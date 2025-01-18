import copy
import itertools
import sys
from copy import deepcopy

import joblib
import torch
import numpy as np
import numpy.typing as npt
import torch.nn.functional as F
from typing import Tuple, Any
import math

from numpy import ndarray
from torch import Tensor, GradScaler, nn, optim
from torch.backends.cudnn import deterministic
from torch.optim import Optimizer

from trackmania_rl import utilities
from trackmania_rl.buffer_management import ReplayBuffer
from config_files import config_copy
from .core import SquashedGaussianMLPActor, MLPQFunction


class MLPActorCritic(nn.Module):

    def __init__(self, hidden_sizes=(256,128),
                 activation=nn.LeakyReLU):
        super().__init__()


        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(hidden_sizes, 3, 1, activation)
        self.q1 = MLPQFunction(hidden_sizes, activation)
        self.q2 = MLPQFunction(hidden_sizes, activation)

    @torch.jit.export
    def act(self, obs: Tuple[torch.Tensor, torch.Tensor], deterministic: bool = False) -> torch.Tensor:

        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a



@torch.compile(disable=not config_copy.is_linux, dynamic=False)
def sac_loss(
        online_network: torch.nn.Module,
        target_network: torch.nn.Module,
        state_img: torch.Tensor,
        state_float: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_state_img: torch.Tensor,
        next_state_float: torch.Tensor,
        gammas: torch.Tensor,
        log_alpha: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

def compile_model(uncompiled_model, jit: bool, is_inference: bool) -> MLPActorCritic:
    if jit:
        if config_copy.is_linux:
            compile_mode = None if "rocm" in torch.__version__ else (
                "max-autotune" if is_inference else "max-autotune-no-cudagraphs")
            model = torch.compile(uncompiled_model, dynamic=False, mode=compile_mode)
        else:
            model = torch.jit.script(uncompiled_model)
    else:
       model = copy.copy(uncompiled_model)
    return model


def make_untrained_sac_network(jit: bool, is_inference: bool) -> Tuple[MLPActorCritic, MLPActorCritic]:
    """
    Constructs two identical copies of the SAC network.

    The first copy is compiled (if jit == True) and is used for inference, for rollouts, for training, etc...
    The second copy is never compiled and **only** used to efficiently share a neural network's weights between processes.

    Args:
        jit: a boolean indicating whether compilation should be used
        is_inference: a boolean indicating whether the model will be used for inference
            (affects compilation optimization mode)

    Returns:
        Tuple containing:
        - The (potentially compiled) model for actual use
        - An uncompiled copy for weight sharing between processes
    """
    uncompiled_model = MLPActorCritic()
    return (
        compile_model(uncompiled_model, jit, is_inference).to(device="cuda", memory_format=torch.channels_last).train(),
        uncompiled_model.to(device="cuda", memory_format=torch.channels_last).train(),
    )


class Trainer:
    def __init__(
            self,
            batch_size: int,
            autolearn_alpha: bool,
            cumul_number_memories_generated: int
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.target_entropy = -3
        self.autolearn_alpha = autolearn_alpha

        self.ac, self.ac_uncompiled = make_untrained_sac_network(config_copy.use_jit, is_inference=False)
        print(f"ac weight location: {self.ac.pi.net[0].weight.data_ptr()}")
        print(f"ac_uncompiled weight location: {self.ac_uncompiled.pi.net[0].weight.data_ptr()}")
        self.ac_targ = deepcopy(self.ac_uncompiled)

        # Disable gradients for target network BEFORE compilation
        for param in self.ac_targ.parameters():
            param.requires_grad_(False)
            param.detach_()  # Explicitly detach parameters

        self.ac_targ = compile_model(self.ac_targ, config_copy.use_jit, is_inference=False)

        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        for param in self.q_params:
            param.requires_grad_(True)


        q_lr = utilities.from_exponential_schedule(
            config_copy.critic_lr_schedule, cumul_number_memories_generated,
        )
        pi_lr = utilities.from_exponential_schedule(
            config_copy.policy_lr_schedule, cumul_number_memories_generated
        )

        weight_decay_lr_ratio = config_copy.weight_decay_lr_ratio

        self.q_optimizer = torch.optim.RAdam([p for name, p in self.ac.named_parameters()
                                             if any(x in name for x in ['q1', 'q2'])],
                                            lr=q_lr,
                                            eps=config_copy.adam_epsilon,
                                            betas=(config_copy.adam_beta1, config_copy.adam_beta2),
                                            weight_decay=q_lr * weight_decay_lr_ratio)
        self.q_scaler = torch.amp.GradScaler(self.device)

        self.pi_optimizer = torch.optim.RAdam([p for name, p in self.ac.named_parameters()
                                             if any(x in name for x in ['pi'])],
                                            lr=pi_lr,
                                            eps=config_copy.adam_epsilon,
                                            betas=(config_copy.adam_beta1, config_copy.adam_beta2),
                                            weight_decay=pi_lr * weight_decay_lr_ratio)

        self.pi_scaler = torch.amp.GradScaler(self.device)


        
        if autolearn_alpha:
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * config_copy.alpha_initial_value).requires_grad_(True)
            self.alpha_scaler = torch.amp.GradScaler(self.device)
            self.alpha_t = torch.exp(self.log_alpha.detach())
        else:
            self.alpha_t = torch.tensor(config_copy.alpha_initial_value).to(self.device)
            self.log_alpha = None
        alpha_lr = utilities.from_exponential_schedule(
            config_copy.alpha_lr_schedule, cumul_number_memories_generated
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                lr=alpha_lr,
                                                eps=config_copy.adam_epsilon,
                                                betas=(config_copy.adam_beta1, config_copy.adam_beta2),
                                                weight_decay=alpha_lr * weight_decay_lr_ratio)

    @staticmethod
    def cleanup_batch(data):
        (o_img, o_float, a, r,
         o2_img, o2_float, gamma, d) = data

        if isinstance(a, torch.Tensor):
            a = a.detach().clone().requires_grad_(True)
        else:
            a = torch.tensor(a, requires_grad=True, device='cuda')

        o = (o_img, o_float)
        o2 = (o2_img, o2_float)

        return o, a, r, o2, d, gamma

    def update(self, data, learn: bool):
        # First run one gradient descent step for Q1 and Q2
        old_weights = self.ac.q1.q[0].weight.clone()
        o, a, r, o2, d, gamma = data

        pi, logp_pi = self.ac.pi(o, deterministic=not learn)

        loss_alpha = None
        if self.autolearn_alpha:
            self.alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()

        alpha_t = self.alpha_t

        if loss_alpha is not None and learn:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        with torch.no_grad():
            a2, logp_a2 = self.ac.pi(o2, deterministic=not learn)

            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)


        loss_q1 = F.smooth_l1_loss(q1, backup)
        loss_q2 = F.smooth_l1_loss(q2, backup)
        loss_q = (loss_q1 + loss_q2) / 2
        print(f"q1={q1[0]}, q2={q2[0]}, backup={backup[0]}")


        if learn:
            self.q_optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(self.q_params, max_norm=10.0)
            self.q_optimizer.step()


        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for param in self.q_params:
            param.requires_grad_(False)

        # Next run one gradient descent step for pi.
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (alpha_t * logp_pi - q_pi).mean()

        if learn:
            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for param in self.q_params:
            param.requires_grad_(True)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(config_copy.polyak)
                p_targ.data.add_((1 - config_copy.polyak) * p.data)

        return loss_q.item(), loss_pi.item(), loss_alpha.item(), alpha_t.item()

    def train_on_batch(self, buffer: ReplayBuffer, do_learn: bool):
        batch, batch_info = buffer.sample(self.batch_size, return_info=True)
        clean_batch = self.cleanup_batch(batch)

        losses = self.update(clean_batch, do_learn)

        if config_copy.prio_alpha > 0:
            buffer.update_priority(batch_info["index"], losses[0].detach().cpu().type(torch.float64))

        return losses

    def save(self, save_dir):
        utilities.save_checkpoint(
            save_dir,
            self.ac,
            self.ac_targ,
            self.pi_optimizer,
            self.q_optimizer,
            self.alpha_optimizer,
            self.pi_scaler,
            self.q_scaler,
            self.alpha_scaler,
        )

    def load_weights_and_stats(self, save_dir):
        # TODO: THIS DOESNT WORK
        try:
            self.ac.load_state_dict(torch.load(f=save_dir / "weights1.torch", weights_only=False))
            self.ac_targ.load_state_dict(torch.load(f=save_dir / "weights2.torch", weights_only=False))
            print(" =====================     Learner weights loaded !     ============================")
        except:
            print(" Learner could not load weights")

        try:
            accumulated_stats = joblib.load(save_dir / "accumulated_stats.joblib")
            print(" =====================      Learner stats loaded !      ============================")
        except:
            accumulated_stats = None
            print(" Learner could not load stats")

        try:
            self.pi_optimizer.load_state_dict(torch.load(f=save_dir / "policy_optimizer.torch", weights_only=False))
            self.q_optimizer.load_state_dict(torch.load(f=save_dir / "q_optimizer.torch", weights_only=False))
            self.alpha_optimizer.load_state_dict(torch.load(f=save_dir / "alpha_optimizer.torch", weights_only=False))
            self.pi_scaler.load_state_dict(torch.load(f=save_dir / "policy_scaler.torch", weights_only=False))
            self.q_scaler.load_state_dict(torch.load(f=save_dir / "q_scaler.torch", weights_only=False))
            self.alpha_scaler.load_state_dict(torch.load(f=save_dir / "alpha_scaler.torch", weights_only=False))
            print(" =========================     Optimizers loaded !     ================================")
        except:
            print(" Could not load optimizer")

        return accumulated_stats


class Inferer:
    """Handles inference for SAC policy network"""
    __slots__ = ("ac", "is_explo")

    def __init__(self, inference_network: MLPActorCritic):
        self.ac = inference_network
        self.is_explo = None

    def infer_network(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray) -> ndarray:
        """Get Q-values for a given state using deterministic policy"""
        state_img_tensor = (
           torch.from_numpy(img_inputs_uint8)
           .unsqueeze(0)
           .to("cuda", memory_format=torch.channels_last, non_blocking=True,
               dtype=torch.float32)
           - 128
        ) / 128
        state_float_tensor = torch.from_numpy(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)

        o = (state_img_tensor, state_float_tensor)

        policy_action, _ = self.ac.pi(o, deterministic=not self.is_explo, with_logprob=False)
        with torch.no_grad():
            q1 = self.ac.q1(o, policy_action)
            q2 = self.ac.q2(o, policy_action)

        return torch.min(q1, q2).detach().cpu().numpy()

    def get_exploration_action(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray) -> tuple[ndarray, ndarray]:
        state_img_tensor = (
           torch.from_numpy(img_inputs_uint8)
           .unsqueeze(0)
           .to("cuda", memory_format=torch.channels_last, non_blocking=True,
               dtype=torch.float32)
           - 128
        ) / 128
        state_float_tensor = torch.from_numpy(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)
        o = (state_img_tensor, state_float_tensor)
        actions = self.ac.act(o, not self.is_explo)
        with torch.no_grad():
            q1 = self.ac.q1(o, actions)
            q2 = self.ac.q2(o, actions)

        return actions.cpu().numpy(), torch.min(q1, q2).detach().cpu().numpy()


