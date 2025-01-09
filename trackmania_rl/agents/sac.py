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
from torch.optim import Optimizer

from trackmania_rl import utilities
from trackmania_rl.buffer_management import ReplayBuffer
from config_files import config_copy


N_ACTIONS = len(["steer", "gas", "break"])

class Normal:
    """Jit workaroung"""
    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        resized_ = torch.broadcast_tensors(loc, scale)
        self.loc = resized_[0]
        self.scale = resized_[1]
        self._batch_shape = list(self.loc.size())

    def _extended_shape(self, sample_shape: list[int]) -> list[int]:
        return sample_shape + self._batch_shape

    def sample(self) -> torch.Tensor:
        return torch.normal(self.loc.expand(self._batch_shape), self.scale.expand(self._batch_shape))

    def rsample(self) -> torch.Tensor:
        epsilon = torch.randn_like(self.scale)
        return self.loc + self.scale * epsilon

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        var = self.variance
        log_scale = torch.log(self.scale)
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))


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
        activation_function = torch.nn.LeakyReLU

        self.LOG_STD_MIN = -5
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

        q_net_builder = lambda: torch.nn.Sequential(
            torch.nn.Linear(dense_hidden_dimension + n_actions, dense_hidden_dimension),
            torch.nn.LayerNorm(dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, dense_hidden_dimension),
            torch.nn.LayerNorm(dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, 1)
        )

        # Q-value networks
        self.q1_net = q_net_builder()
        self.q2_net = q_net_builder()

        self.register_buffer('float_inputs_mean', float_inputs_mean)
        self.register_buffer('float_inputs_std', float_inputs_std)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        for module in [self.img_head, self.float_feature_extractor, self.shared_net, self.q1_net, self.q2_net]:
            for m in module:
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
        return state_img_tensor, state_float_tensor

    def get_features(self, img: torch.Tensor, float_inputs: torch.Tensor) -> torch.Tensor:
        """Get features using preprocessed inputs"""
        img, float_inputs = self.preprocess_inputs(img, float_inputs)

        img_outputs = self.img_head(img)
        float_outputs = self.float_feature_extractor(float_inputs)
        features = torch.cat((img_outputs, float_outputs), 1)
        return self.shared_net(features)

    def forward(self, img: torch.Tensor, float_inputs: torch.Tensor,
                deterministic: bool, with_logprob: bool) -> tuple[Tensor, Tensor | None]:
        features = self.get_features(img, float_inputs)
        # Get steering distribution parameters
        mean = self.policy_mean(features)
        log_std = self.policy_log_std(features)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()

        if torch.isnan(mean).any() or torch.isnan(std).any():
            print(features)
            print(mean)
            print(log_std)
            print(std)
            raise ValueError("NaN detected in policy distribution parameters")

        policy_distribution = Normal(mean, std)
        if deterministic:
            policy_action = mean
        else:
            policy_action = policy_distribution.rsample()

        policy_action = torch.tanh(policy_action)

        if with_logprob:
            policy_logprob = policy_distribution.log_prob(policy_action)
            policy_logprob -= torch.log(1 - policy_action.pow(2) + 1e-6)  # so no log of 0
            policy_logprob = policy_logprob.sum(1, keepdim=True)
        else:
            policy_logprob = None

        return policy_action, policy_logprob

    @torch.jit.export
    def q_values(self, img: torch.Tensor, float_inputs: torch.Tensor, actions: torch.Tensor):
        features = self.get_features(img, float_inputs)
        # Create new tensor for concatenation instead of modifying in place
        sa = torch.cat([features, actions], dim=-1)
        q1 = self.q1_net(sa).squeeze()
        q2 = self.q2_net(sa).squeeze()
        return q1, q2

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self



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


class Trainer:
    def __init__(
            self,
            online_network: SAC_Network,
            target_network: SAC_Network,
            policy_optimizer: Optimizer,
            critic_optimizer: Optimizer,
            alpha_optimizer: Optimizer,
            policy_scaler: GradScaler,
            critic_scaler: GradScaler,
            alpha_scaler: GradScaler,
            batch_size: int,
            log_alpha: torch.Tensor,
    ):
        self.online_network = online_network
        self.target_network = target_network
        self.policy_optimizer = policy_optimizer
        self.q_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.policy_scaler = policy_scaler
        self.q_scaler = critic_scaler
        self.alpha_scaler = alpha_scaler
        self.batch_size = batch_size
        self.log_alpha = log_alpha
        self.train_steps = 0

        self.scaler = torch.amp.GradScaler('cuda')
        self.max_grad_norm = 1.0

    def check_nan(self, tensor: torch.Tensor, name: str):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            raise ValueError(f"NaN in {name}")

    # In the Trainer class __init__
    def __init__(
            self,
            online_network: torch.nn.Module,
            target_network: torch.nn.Module,
            policy_optimizer: Optimizer,
            critic_optimizer: Optimizer,
            alpha_optimizer: Optimizer,
            policy_scaler: GradScaler,
            critic_scaler: GradScaler,
            alpha_scaler: GradScaler,
            batch_size: int,
            log_alpha: torch.Tensor,
    ):
        self.online_network = online_network
        self.target_network = target_network
        self.policy_optimizer = policy_optimizer
        self.q_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.policy_scaler = policy_scaler
        self.q_scaler = critic_scaler
        self.alpha_scaler = alpha_scaler
        self.batch_size = batch_size
        self.log_alpha = log_alpha
        self.train_steps = 0
        self.max_grad_norm = 1.0

    def train_on_batch(self, buffer: ReplayBuffer, do_learn: bool):
        target_entropy = -N_ACTIONS

        batch, batch_info = buffer.sample(self.batch_size, return_info=True)
        (state_img_tensor, state_float_tensor, actions, rewards,
         next_state_img_tensor, next_state_float_tensor, gamma, done) = batch

        alpha_t = 0.2
        #alpha_t = torch.exp(self.log_alpha.detach())
        
        with torch.no_grad():
            next_policy_actions, next_policy_logprob = self.online_network(next_state_img_tensor, next_state_float_tensor, deterministic=not do_learn, with_logprob=True)
            next_q1, next_q2 = self.target_network.q_values(next_state_img_tensor, next_state_float_tensor, next_policy_actions)
            min_next_q = (torch.min(next_q1, next_q2) - alpha_t * next_policy_logprob.squeeze())
            next_q_value = rewards + (1 - done) * gamma * min_next_q.view(-1)

        q1, q2 = self.online_network.q_values(state_img_tensor, state_float_tensor, actions)
        q1_loss, q2_loss = F.mse_loss(q1, next_q_value), F.mse_loss(q2, next_q_value)
        loss_q = q1_loss + q2_loss

        if do_learn:
            self.q_optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), self.max_grad_norm)
            self.q_optimizer.step()

        policy_actions, policy_logprob = self.online_network(state_img_tensor, state_float_tensor, deterministic=not do_learn, with_logprob=True)
        q1, q2 = self.online_network.q_values(state_img_tensor, state_float_tensor, policy_actions)
        q = torch.min(q1, q2)
        loss_policy = (q - (alpha_t * policy_logprob)).mean()

        #loss_alpha = (-self.log_alpha.exp() * (policy_logprob + target_entropy)).mean()

        if do_learn:
            #policy_alpha_loss = loss_alpha + loss_policy
            self.policy_optimizer.zero_grad()
            #self.alpha_optimizer.zero_grad()
            #policy_alpha_loss.backward()
            loss_policy.backward()
            torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            #self.alpha_optimizer.step()

            with torch.no_grad():
                for param, param_targ in zip(self.online_network.parameters(), self.target_network.parameters()):
                    param_targ.data.mul_(config_copy.polyak)
                    param_targ.data.add_((1 - config_copy.polyak) * param.data)

        return loss_q.item(), loss_policy.item(), loss_alpha.item(), alpha_t.item()


class Inferer:
    """Handles inference for SAC policy network"""
    __slots__ = ("inference_network", "is_explo")

    def __init__(self, inference_network: torch.nn.Module):
        self.inference_network = inference_network
        self.is_explo = None

    def infer_network(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray) -> ndarray:
        """Get Q-values for a given state using deterministic policy"""
        img_inputs_uint8 = torch.Tensor(img_inputs_uint8).unsqueeze(0)
        float_inputs = torch.Tensor(float_inputs).unsqueeze(0)

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

    def get_exploration_action(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray) -> tuple[ndarray, ndarray]:
        img_inputs_uint8 = torch.Tensor(img_inputs_uint8).unsqueeze(0)
        float_inputs = torch.Tensor(float_inputs).unsqueeze(0)

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

    float_inputs_mean = torch.tensor(config_copy.float_inputs_mean, dtype=torch.float32)
    float_inputs_std = torch.tensor(config_copy.float_inputs_std, dtype=torch.float32)

    uncompiled_model = SAC_Network(
        float_inputs_dim=config_copy.float_input_dim,
        float_hidden_dim=config_copy.float_hidden_dim,
        conv_head_output_dim=config_copy.conv_head_output_dim,
        dense_hidden_dimension=config_copy.dense_hidden_dimension,
        n_actions=N_ACTIONS,
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