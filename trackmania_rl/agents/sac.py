import copy
import itertools
import sys
import torch
import numpy as np
import numpy.typing as npt
import torch.nn.functional as F
from typing import Tuple, Any
import math

from numpy import ndarray
from torch import Tensor, GradScaler
from torch.backends.cudnn import deterministic
from torch.distributions import Categorical
from torch.optim import Optimizer

from trackmania_rl import utilities
from trackmania_rl.buffer_management import ReplayBuffer
from config_files import config_copy


N_ACTIONS = len(["left", "right", "gas", "break"])

class SAC_Network(torch.nn.Module):
    def __init__(
            self,
            float_inputs_dim: int,
            float_hidden_dim: int,
            conv_head_output_dim: int,
            dense_hidden_dimension: int,
            n_actions: int,
            float_inputs_mean: torch.Tensor,
            float_inputs_std: torch.Tensor,
    ):
        super().__init__()
        self.dense_hidden_dimension = dense_hidden_dimension
        self.dense_input_dimension = conv_head_output_dim + float_hidden_dim
        activation_function = torch.nn.LeakyReLU

        self.LOG_STD_MIN = -5
        self.LOG_STD_MAX = 2
        self.EPS = 1e-7

        # Image processing head
        img_head_channels = [1, 16, 32, 64, 32]
        img_head_builder = lambda: torch.nn.Sequential(
            torch.nn.Conv2d(img_head_channels[0], img_head_channels[1], kernel_size=(4, 4), stride=2),
            torch.nn.BatchNorm2d(img_head_channels[1]),
            activation_function(inplace=True),
            torch.nn.Conv2d(img_head_channels[1], img_head_channels[2], kernel_size=(4, 4), stride=2),
            torch.nn.BatchNorm2d(img_head_channels[2]),
            activation_function(inplace=True),
            torch.nn.Conv2d(img_head_channels[2], img_head_channels[3], kernel_size=(3, 3), stride=2),
            torch.nn.BatchNorm2d(img_head_channels[3]),
            activation_function(inplace=True),
            torch.nn.Conv2d(img_head_channels[3], img_head_channels[4], kernel_size=(3, 3), stride=1),
            torch.nn.BatchNorm2d(img_head_channels[4]),
            activation_function(inplace=True),
            torch.nn.Flatten(),
        )

        self.policy_img_head = img_head_builder()
        self.q_img_head = img_head_builder()

        float_feature_extractor_builder = lambda: torch.nn.Sequential(
            torch.nn.Linear(float_inputs_dim, float_hidden_dim),
            torch.nn.LayerNorm(float_hidden_dim),
            activation_function(inplace=True),
            torch.nn.Linear(float_hidden_dim, float_hidden_dim),
            torch.nn.LayerNorm(float_hidden_dim),
            activation_function(inplace=True),
        )

        # Float feature processing
        self.policy_float_feature_extractor = float_feature_extractor_builder()
        self.q_float_feature_extractor = float_feature_extractor_builder()

        shared_net_builder = lambda: torch.nn.Sequential(
            torch.nn.Linear(self.dense_input_dimension, dense_hidden_dimension),
            torch.nn.LayerNorm(dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, dense_hidden_dimension),
            torch.nn.LayerNorm(dense_hidden_dimension),
            activation_function(inplace=True),
        )


        # Shared feature network
        self.policy_shared_net = shared_net_builder()
        self.q_shared_net = shared_net_builder()

        # Policy heads
        self.action_logits = torch.nn.Linear(dense_hidden_dimension, n_actions)

        q_net_builder = lambda: torch.nn.Sequential(
            torch.nn.Linear(dense_hidden_dimension, dense_hidden_dimension),
            torch.nn.LayerNorm(dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, dense_hidden_dimension),
            torch.nn.LayerNorm(dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, n_actions)
        )

        # Q-value networks
        self.q1_net = q_net_builder()
        self.q2_net = q_net_builder()

        self.q1_target_net = q_net_builder()
        self.q2_target_net = q_net_builder()

        self.q1_target_net.load_state_dict(self.q1_net.state_dict())
        self.q2_target_net.load_state_dict(self.q2_net.state_dict())


        self.float_inputs_mean = float_inputs_mean.clone().detach().to(dtype=torch.float32, device="cuda")
        self.float_inputs_std = float_inputs_std.clone().detach().to(dtype=torch.float32, device="cuda")

        self.initialize_weights()

    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.weight, 1.0)
                torch.nn.init.constant_(m.bias, 0.0)

    def preprocess_inputs(self, img: torch.Tensor, float_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state_img_tensor = img.to(device="cuda",
                                  dtype=torch.float32,
                                  memory_format=torch.channels_last,
                                  non_blocking=True)
        state_img_tensor = ((state_img_tensor.float() - 128) / 128)


        state_float_tensor = float_inputs.to(device="cuda",
                                             dtype=torch.float32,
                                             non_blocking=True)
        return state_img_tensor, (state_float_tensor - self.float_inputs_mean) / self.float_inputs_std

    def forward(self, img: torch.Tensor, float_inputs: torch.Tensor, deterministic: bool) -> tuple[Tensor, Tensor, Tensor]:
        img, float_inputs = self.preprocess_inputs(img, float_inputs)
        img_outputs = self.policy_img_head(img)
        float_outputs = self.policy_float_feature_extractor(float_inputs)
        features = self.policy_shared_net(torch.cat((img_outputs, float_outputs), dim=-1))

        # Get steering distribution parameters
        logits = self.action_logits(features)
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = torch.argmax(probs).squeeze(-1)
        else:
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        log_prob = F.log_softmax(logits, dim=-1) #Check if dim=1 gives same results

        return action, log_prob, probs

    def q_get_features(self, img: torch.Tensor, float_inputs: torch.Tensor) -> torch.Tensor:
        """Get features using preprocessed inputs"""
        img, float_inputs = self.preprocess_inputs(img, float_inputs)
        img_outputs = self.q_img_head(img)
        float_outputs = self.q_float_feature_extractor(float_inputs)
        return self.q_shared_net(torch.cat((img_outputs, float_outputs), dim=-1))

    @torch.jit.export
    def q_values(self, img: torch.Tensor, float_inputs: torch.Tensor):
        features = self.q_get_features(img, float_inputs)
        q1 = self.q1_net(features)
        q2 = self.q2_net(features)
        return q1, q2

    @torch.jit.export
    def target_q_values(self, img: torch.Tensor, float_inputs: torch.Tensor):
        features = self.q_get_features(img, float_inputs)
        q1 = self.q1_target_net(features)
        q2 = self.q2_target_net(features)
        return q1, q2

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self



@torch.compile(disable=not config_copy.is_linux, dynamic=False)
def sac_loss(
        network: torch.nn.Module,
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


class Trainer:
    def __init__(
            self,
            network: SAC_Network,
            policy_optimizer: Optimizer,
            critic_optimizer: Optimizer,
            alpha_optimizer: Optimizer,
            policy_scaler: GradScaler,
            critic_scaler: GradScaler,
            alpha_scaler: GradScaler,
            batch_size: int,
            log_alpha: torch.Tensor,
    ):
        self.network = network
        self.policy_optimizer = policy_optimizer
        self.q_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.policy_scaler = policy_scaler
        self.q_scaler = critic_scaler
        self.alpha_scaler = alpha_scaler
        self.batch_size = batch_size
        self.log_alpha = log_alpha
        self.train_steps = 0
        self.max_grad_norm = 1

    @torch.compile(disable=not config_copy.is_linux, dynamic=False)
    def train_on_batch(self, buffer: ReplayBuffer, do_learn: bool):
        target_entropy = -0.89 *  torch.log(1 / torch.tensor(N_ACTIONS))

        batch, batch_info = buffer.sample(self.batch_size, return_info=True)
        (state_img_tensor, state_float_tensor, actions, rewards,
         next_state_img_tensor, next_state_float_tensor, gamma, done) = batch

        if actions.dtype != torch.int64:
            actions = actions.to(dtype=torch.int64)

        #alpha_t = 0.2
        alpha_t = torch.exp(self.log_alpha.detach())
        
        with torch.no_grad():
            _, next_policy_logprob, next_policy_action_probs = self.network(next_state_img_tensor, next_state_float_tensor, deterministic=not do_learn)
            next_target_q1, next_target_q2 = self.network.target_q_values(next_state_img_tensor, next_state_float_tensor)
            min_next_target_q = next_policy_action_probs * (torch.min(next_target_q1, next_target_q2) - alpha_t * next_policy_logprob)
            min_next_target_q = min_next_target_q.sum(dim=1)
            next_q_value = rewards + (1 - done) * gamma * min_next_target_q.view(-1)

        q1, q2 = self.network.q_values(state_img_tensor, state_float_tensor)
        q1_action_value, q2_action_value = q1.gather(1, actions.unsqueeze(-1)).squeeze(), q2.gather(1, actions.unsqueeze(-1)).squeeze()
        q1_loss, q2_loss = F.mse_loss(q1_action_value, next_q_value), F.mse_loss(q2_action_value, next_q_value)
        print(f"next_q_value={next_q_value[0].item()}, q1_action_value={q1_action_value[0].item()}, q2_action_value={q1_action_value[0].item()}")
        loss_q = q1_loss + q2_loss

        if do_learn:
            self.q_optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.q_optimizer.step()

        _, policy_logprob, policy_action_probs = self.network(state_img_tensor, state_float_tensor, deterministic=not do_learn)
        with torch.no_grad():
            q1, q2 = self.network.q_values(state_img_tensor, state_float_tensor)
            q = torch.min(q1, q2)
        loss_policy = (policy_action_probs * ((alpha_t * policy_logprob) - q)).mean()

        #loss_alpha = (-self.log_alpha.exp() * (policy_logprob + target_entropy)).mean()

        if do_learn:
            # policy_alpha_loss = loss_alpha + loss_policy
            self.policy_optimizer.zero_grad()
            # self.alpha_optimizer.zero_grad()
            # policy_alpha_loss.backward()
            loss_policy.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            # self.alpha_optimizer.step()

        loss_alpha = (policy_action_probs.detach() * (-self.log_alpha.exp() * (policy_logprob + target_entropy).detach())).mean()

        if do_learn:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.alpha_optimizer.step()

            with torch.no_grad():
                for param, param_targ in zip(
                        (*self.network.q1_net.parameters(), *self.network.q2_net.parameters()), (*self.network.q1_target_net.parameters(), *self.network.q2_net.parameters())
                ):
                    param_targ.data.mul_(config_copy.polyak)
                    param_targ.data.add_((1 - config_copy.polyak) * param.data)

        return loss_q.item(), loss_policy.item(), loss_alpha.item(), alpha_t.item()


class Inferer:
    """Handles inference for SAC policy network"""
    __slots__ = ("inference_network", "is_explo")

    def __init__(self, inference_network: SAC_Network):
        self.inference_network = inference_network
        self.is_explo = None

    def infer_network(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray) -> ndarray:
        """Get Q-values for a given state using deterministic policy"""
        img_inputs_uint8 = torch.Tensor(img_inputs_uint8).unsqueeze(0)
        float_inputs = torch.Tensor(float_inputs).unsqueeze(0)

        with torch.no_grad():
            q1, q2 = self.inference_network.q_values(
                img_inputs_uint8,
                float_inputs
            )

            return torch.min(q1, q2).cpu().numpy()

    def get_exploration_action(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray) -> tuple[ndarray, ndarray, ndarray]:
        q_values = self.infer_network(img_inputs_uint8, float_inputs)
        img_inputs_uint8 = torch.Tensor(img_inputs_uint8).unsqueeze(0)
        float_inputs = torch.Tensor(float_inputs).unsqueeze(0)

        with torch.no_grad():
            policy_action, _, policy_action_probs = self.inference_network(
                img_inputs_uint8,
                float_inputs,
                not self.is_explo
            )
        return policy_action.item(), np.max(q_values), q_values


def make_untrained_sac_network(jit: bool, is_inference: bool) -> Tuple[SAC_Network, SAC_Network]:
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

    float_inputs_mean = torch.tensor(config_copy.float_inputs_mean, dtype=torch.float32)
    float_inputs_std = torch.tensor(config_copy.float_inputs_std, dtype=torch.float32)

    uncompiled_model = SAC_Network(
        float_inputs_dim=config_copy.float_input_dim,
        float_hidden_dim=config_copy.float_hidden_dim,
        conv_head_output_dim=config_copy.conv_head_output_dim,
        dense_hidden_dimension=config_copy.dense_hidden_dimension,
        n_actions=len(config_copy.inputs),
        float_inputs_mean=float_inputs_mean,
        float_inputs_std=float_inputs_std,
    )

    if jit:
        if config_copy.is_linux:
            # Use different compilation modes for inference vs training
            compile_mode = None if "rocm" in torch.__version__ else (
                "max-autotune" if is_inference else "max-autotune-no-cudagraphs"
            )
            model = torch.compile(uncompiled_model, dynamic=False, mode=compile_mode)
        else:
            #model = torch.compile(uncompiled_model, dynamic=False, mode="reduce-overhead")
            model = torch.jit.script(uncompiled_model)
    else:
        model = copy.deepcopy(uncompiled_model)

    return (
        model.to(device="cuda", memory_format=torch.channels_last).train(),
        uncompiled_model.to(device="cuda", memory_format=torch.channels_last).train(),
    )