import copy
import sys

import torch
import numpy as np
import numpy.typing as npt
import torch.nn.functional as F
from typing import Tuple, Any
import math

from numpy import ndarray
from torch import Tensor
from torch.optim import Optimizer
from trackmania_rl.buffer_management import ReplayBuffer
from config_files import config_copy


N_ACTIONS = len(["steer", "gas", "break"])

class SAC_Network(torch.nn.Module):
    def __init__(
            self,
            float_inputs_dim: int,
            float_hidden_dim: int,
            conv_head_output_dim: int,
            dense_hidden_dimension: int,
            n_actions: int,
            float_inputs_mean: npt.NDArray,
            float_inputs_std: npt.NDArray,
    ):
        super().__init__()
        self.dense_hidden_dimension = dense_hidden_dimension
        activation_function = torch.nn.LeakyReLU

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.EPS = 1e-7

        # Image processing head
        img_head_channels = [1, 16, 32, 64, 32]
        self.img_head = torch.nn.Sequential(
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

        # Float feature processing
        self.float_feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(float_inputs_dim, float_hidden_dim),
            torch.nn.LayerNorm(float_hidden_dim),
            activation_function(inplace=True),
            torch.nn.Linear(float_hidden_dim, float_hidden_dim),
            torch.nn.LayerNorm(float_hidden_dim),
            activation_function(inplace=True),
        )

        # Combined features dimension
        self.dense_input_dimension = conv_head_output_dim + float_hidden_dim

        # Shared feature network
        self.shared_net = torch.nn.Sequential(
            torch.nn.Linear(self.dense_input_dimension, dense_hidden_dimension),
            torch.nn.LayerNorm(dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, dense_hidden_dimension),
            torch.nn.LayerNorm(dense_hidden_dimension),
            activation_function(inplace=True),
        )

        # Policy heads
        self.policy_mean = torch.nn.Linear(dense_hidden_dimension, n_actions)
        self.policy_log_std = torch.nn.Linear(dense_hidden_dimension, n_actions)

        # Q-value networks
        self.q1_net = torch.nn.Sequential(
            torch.nn.Linear(dense_hidden_dimension + n_actions, dense_hidden_dimension),
            torch.nn.LayerNorm(dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, dense_hidden_dimension),
            torch.nn.LayerNorm(dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, 1)
        )

        self.q2_net = torch.nn.Sequential(
            torch.nn.Linear(dense_hidden_dimension + n_actions, dense_hidden_dimension),
            torch.nn.LayerNorm(dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, dense_hidden_dimension),
            torch.nn.LayerNorm(dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, 1)
        )

        self.float_inputs_mean = torch.tensor(float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(float_inputs_std, dtype=torch.float32).to("cuda")

        self.initialize_weights()
        self.steer_entropy_scale = 1.0
        self.discrete_entropy_scale = 0.1

    def initialize_weights(self) -> None:
        # Initialize convolutional layers
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)

            # Special initialization for policy heads
        for module in [self.policy_mean, self.policy_log_std]:
            torch.nn.init.uniform_(module.weight, -3e-3, 3e-3)
            torch.nn.init.uniform_(module.bias, -3e-3, 3e-3)

            # Special initialization for final Q-network layers
        for q_net in [self.q1_net, self.q2_net]:
            final_layer = q_net[-1]
            torch.nn.init.uniform_(final_layer.weight, -3e-3, 3e-3)
            torch.nn.init.uniform_(final_layer.bias, -3e-3, 3e-3)

    def preprocess_inputs(self, img: npt.NDArray | torch.Tensor, float_inputs: npt.NDArray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).unsqueeze(0)
            float_inputs = torch.from_numpy(float_inputs).unsqueeze(0)

        state_img_tensor = img.to("cuda", memory_format=torch.channels_last, non_blocking=True)
        state_img_tensor = ((state_img_tensor.float() - 128) / 128)

        state_float_tensor = float_inputs.to("cuda", non_blocking=True)
        return state_img_tensor, state_float_tensor

    def get_features(self, img: npt.NDArray | torch.Tensor, float_inputs: npt.NDArray | torch.Tensor) -> torch.Tensor:
        """Get features using preprocessed inputs"""
        img, float_inputs = self.preprocess_inputs(img, float_inputs)

        img_outputs = self.img_head(img)
        float_outputs = self.float_feature_extractor(float_inputs)
        features = torch.cat((img_outputs, float_outputs), 1)
        return self.shared_net(features)

    def forward(self, img: npt.NDArray | torch.Tensor, float_inputs: npt.NDArray | torch.Tensor,
                deterministic: bool, with_logprob: bool) -> tuple[Tensor, Tensor | None]:
        features = self.get_features(img, float_inputs)

        # Get steering distribution parameters
        mean = self.policy_mean(features)
        mean = torch.clamp(mean, -3.0, 3.0)
        log_std = torch.clamp(self.policy_log_std(features), self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        policy_distribution = torch.distributions.Normal(mean, std)
        if deterministic:
            policy_action = mean
        else:
            policy_action = policy_distribution.sample()

        if with_logprob:
            policy_logprob = policy_distribution.log_prob(policy_action).sum(axis=-1)
            policy_logprob -= (2 * (np.log(2) - policy_action - F.softplus(-2 * policy_action))).sum(axis=1)
        else:
            policy_logprob = None

        policy_action = torch.tanh(policy_action)


        return policy_action, policy_logprob

    @torch.jit.export
    def q_values(self, img, float_inputs, actions):
        features = self.get_features(img, float_inputs)
        # Create new tensors instead of inplace operations
        sa = torch.cat([features.detach().clone(), actions], dim=-1)

        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)

        return q1, q2

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self


#@torch.compile(disable=not config_copy.is_linux, dynamic=False)
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
        target_entropy = -N_ACTIONS

        # Get current policy action and log prob
        policy_action, policy_logprob = online_network(state_img, state_float,
                                                       deterministic=False, with_logprob=True)

        # Compute temperature parameter
        alpha = torch.exp(log_alpha.detach())

        # Alpha loss
        alpha_loss = -(log_alpha * (policy_logprob + target_entropy).detach()).mean()

        # Current Q-values
        q1, q2 = online_network.q_values(state_img, state_float, actions)

        # Compute target Q-values
        with torch.no_grad():
            next_policy_action, next_policy_logprob = online_network(
                next_state_img, next_state_float,
                deterministic=False, with_logprob=True
            )

            # Get target Q-values
            q1_target, q2_target = target_network.q_values(
                next_state_img, next_state_float,
                next_policy_action
            )
            q_target = torch.min(q1_target, q2_target)

            # Compute backup (target value)
            done = torch.zeros_like(rewards)  # If done tensor isn't provided
            backup = rewards + gammas * (1 - done) * (q_target - alpha * next_policy_logprob)

        # Compute critic losses
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        critic_loss = (loss_q1 + loss_q2) / 2

        # Compute policy Q-values
        q1_policy, q2_policy = online_network.q_values(state_img, state_float, policy_action)
        q_policy = torch.min(q1_policy, q2_policy)

        # Entropy-regularized policy loss
        policy_loss = (alpha * policy_logprob - q_policy).mean()

        return critic_loss, policy_loss, alpha_loss


class Trainer:
    def __init__(
            self,
            online_network: torch.nn.Module,
            target_network: torch.nn.Module,
            policy_optimizer: Optimizer,
            critic_optimizer: Optimizer,
            alpha_optimizer: Optimizer,
            batch_size: int,
            log_alpha: torch.Tensor,
    ):
        self.online_network = online_network
        self.target_network = target_network
        self.policy_optimizer = policy_optimizer
        self.q_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.batch_size = batch_size
        self.log_alpha = log_alpha
        self.train_steps = 0

    def train_on_batch(self, buffer: ReplayBuffer, do_learn: bool):
        target_entropy = -N_ACTIONS
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            # Get batch
            batch, batch_info = buffer.sample(self.batch_size, return_info=True)
            (state_img_tensor, state_float_tensor, actions, rewards,
             next_state_img_tensor, next_state_float_tensor, gamma, done) = batch

            policy_action, policy_logprob = self.online_network(state_img_tensor, state_float_tensor, deterministic=not do_learn, with_logprob=True)

            alpha_t = torch.exp(self.log_alpha.detach())
            alpha_loss = -(self.log_alpha * (policy_logprob + target_entropy).detach()).mean()

            if do_learn:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            q1, q2 = self.online_network.q_values(state_img_tensor, state_float_tensor, policy_action)

            with torch.no_grad():
                next_policy_action, next_policy_logprob = self.online_network(next_state_img_tensor, next_state_float_tensor, deterministic=not do_learn, with_logprob=True)
                q1_target, q2_target = self.target_network.q_values(next_state_img_tensor, next_state_float_tensor, next_policy_action)
                q_target = torch.min(q1_target, q2_target)
                backup = rewards + gamma * (1 - done) * (q_target - alpha_t * next_policy_logprob)

            loss_q1 = ((q1 - backup) ** 2).mean()
            loss_q2 = ((q2 - backup) ** 2).mean()
            q_loss = (loss_q1 + loss_q2) / 2

            if do_learn:
                self.q_optimizer.zero_grad()
                q_loss.backward()
                self.q_optimizer.step()

            self.online_network.q1_net.requires_grad_(False)
            self.online_network.q2_net.requires_grad_(False)

            q1_policy, q2_policy = self.online_network.q_values(state_img_tensor, state_float_tensor, policy_action)
            q_policy = torch.min(q1_policy, q2_policy)

            # Entropy-regularized policy loss
            policy_loss = (alpha_t * policy_logprob - q_policy).mean()

            if do_learn:
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            self.online_network.q1_net.requires_grad_(True)
            self.online_network.q2_net.requires_grad_(True)

            if do_learn:
                with torch.no_grad():
                    for policy, policy_targ in zip(self.online_network.parameters(), self.target_network.parameters()):
                        # NB: We use an in-place operations "mul_", "add_" to update target
                        # params, as opposed to "mul" and "add", which would make new tensors.
                        policy_targ.data.mul_(config_copy.polyak)
                        policy_targ.data.add_((1 - config_copy.polyak) * policy.data)

        return q_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_t.item()


class Inferer:
    """Handles inference for SAC policy network"""
    __slots__ = ("inference_network", "is_explo")

    def __init__(self, inference_network: torch.nn.Module):
        self.inference_network = inference_network
        self.is_explo = None

    def infer_network(self, img_inputs_uint8: np.ndarray, float_inputs: np.ndarray) -> np.ndarray:
        """Get Q-values for a given state using deterministic policy"""
        with torch.no_grad():
            policy_action, _ = self.inference_network(
                img_inputs_uint8,
                float_inputs,
                deterministic=True,
                with_logprob=False
            )

            q1, q2 = self.inference_network.q_values(
                img_inputs_uint8,
                float_inputs,
                policy_action
            )

            return torch.min(q1, q2).cpu().numpy()

    def get_exploration_action(self, img_inputs_uint8: np.ndarray, float_inputs: np.ndarray) -> tuple[ndarray, ndarray]:
        with torch.no_grad():
            policy_action, _ = self.inference_network(
                img_inputs_uint8,
                float_inputs,
                deterministic=False,
                with_logprob=False
            )

            q1, q2 = self.inference_network.q_values(
                img_inputs_uint8,
                float_inputs,
                policy_action
            )

            return policy_action.cpu().numpy(), torch.min(q1, q2).cpu().numpy()


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
    uncompiled_model = SAC_Network(
        float_inputs_dim=config_copy.float_input_dim,
        float_hidden_dim=config_copy.float_hidden_dim,
        conv_head_output_dim=config_copy.conv_head_output_dim,
        dense_hidden_dimension=config_copy.dense_hidden_dimension,
        n_actions=N_ACTIONS,
        float_inputs_mean=config_copy.float_inputs_mean,
        float_inputs_std=config_copy.float_inputs_std,
    )

    if jit:
        if config_copy.is_linux:
            # Use different compilation modes for inference vs training
            compile_mode = None if "rocm" in torch.__version__ else (
                "max-autotune" if is_inference else "max-autotune-no-cudagraphs"
            )
            model = torch.compile(uncompiled_model, dynamic=False, mode=compile_mode)
        else:
            model = torch.jit.script(uncompiled_model)
    else:
        model = copy.deepcopy(uncompiled_model)

    return (
        model.to(device="cuda", memory_format=torch.channels_last).train(),
        uncompiled_model.to(device="cuda", memory_format=torch.channels_last).train(),
    )