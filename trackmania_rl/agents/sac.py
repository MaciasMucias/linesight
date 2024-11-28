import copy
from typing import Tuple
import random

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torchrl.data import ReplayBuffer

from config_files import config_copy
from trackmania_rl import utilities

class SAC_Network(torch.nn.Module):
    def __init__(
            self,
            float_inputs_dim: int,
            float_hidden_dim: int,
            conv_head_output_dim: int,
            dense_hidden_dimension: int,
            float_inputs_mean: npt.NDArray,
            float_inputs_std: npt.NDArray,
    ):
        n_actions = 3 #TODO: move this to config?

        super().__init__()
        activation_function = torch.nn.LeakyReLU

        # Image and float feature processing remains the same
        img_head_channels = [1, 16, 32, 64, 32]
        self.img_head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=img_head_channels[0], out_channels=img_head_channels[1], kernel_size=(4, 4),
                            stride=2),
            activation_function(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[1], out_channels=img_head_channels[2], kernel_size=(4, 4),
                            stride=2),
            activation_function(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[2], out_channels=img_head_channels[3], kernel_size=(3, 3),
                            stride=2),
            activation_function(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[3], out_channels=img_head_channels[4], kernel_size=(3, 3),
                            stride=1),
            activation_function(inplace=True),
            torch.nn.Flatten(),
        )

        self.float_feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(float_inputs_dim, float_hidden_dim),
            activation_function(inplace=True),
            torch.nn.Linear(float_hidden_dim, float_hidden_dim),
            activation_function(inplace=True),
        )

        dense_input_dimension = conv_head_output_dim + float_hidden_dim

        # Shared feature network
        self.shared_features = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension, dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, dense_hidden_dimension),
            activation_function(inplace=True),
        )

        # Continuous action (steering) policy heads
        self.steer_mean = torch.nn.Linear(dense_hidden_dimension, 1)
        self.steer_log_std = torch.nn.Linear(dense_hidden_dimension, 1)

        # Discrete action policy heads
        self.up_logits = torch.nn.Linear(dense_hidden_dimension, 1)
        self.down_logits = torch.nn.Linear(dense_hidden_dimension, 1)

        # Q-networks for hybrid action space
        self.q1_net = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension + n_actions, dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, 1)
        )

        self.q2_net = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension + n_actions, dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, 1)
        )

        self.initialize_weights()

        self.n_actions = n_actions
        # State normalization
        self.float_inputs_mean = torch.tensor(float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(float_inputs_std, dtype=torch.float32).to("cuda")

    def initialize_weights(self):
        lrelu_neg_slope = 1e-2
        activation_gain = torch.nn.init.calculate_gain("leaky_relu", lrelu_neg_slope)

        # Initialize shared feature extractors
        for module in [self.img_head, self.float_feature_extractor, self.shared_features]:
            for m in module:
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                    utilities.init_orthogonal(m, activation_gain)

        # Initialize Q networks
        for module in [self.q1_net, self.q2_net]:
            for m in module:
                if isinstance(m, torch.nn.Linear):
                    utilities.init_orthogonal(m, activation_gain)

        # Initialize policy heads
        utilities.init_orthogonal(self.steer_mean)
        utilities.init_orthogonal(self.steer_log_std)
        utilities.init_orthogonal(self.up_logits)
        utilities.init_orthogonal(self.down_logits)

    def forward(self, img: torch.Tensor, float_inputs: torch.Tensor) -> tuple[
        tuple[tuple[Tensor, Tensor], Tensor, Tensor], Tensor]:
        """
        Forward pass through the SAC network.
        Returns policy distribution and Q-values.
        """
        img_outputs = self.img_head(img)
        float_outputs = self.float_feature_extractor((float_inputs - self.float_inputs_mean) / self.float_inputs_std)
        features = torch.cat((img_outputs, float_outputs), 1)

        # Get shared features
        shared = self.shared_features(features)

        # Continuous steering policy
        steer_mean = torch.tanh(self.steer_mean(shared))  # Bound to [-1, 1]
        steer_log_std = torch.clamp(self.steer_log_std(shared), -20, 2)

        return ((steer_mean, steer_log_std), self.up_logits(shared), self.down_logits(shared)), features

    @torch.jit.export
    def get_q_values(self, features: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values for state-action pairs"""
        sa_pairs = torch.cat([features, actions], dim=1)
        return self.q1_net(sa_pairs), self.q2_net(sa_pairs)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

@torch.compile(disable=not config_copy.is_linux, dynamic=False)
def sac_loss(q1_pred: torch.Tensor, q2_pred: torch.Tensor, q_target: torch.Tensor,
             steer_log_prob: torch.Tensor, up_log_prob: torch.Tensor, down_log_prob: torch.Tensor,
             alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute SAC losses for hybrid action space"""
    # Critic loss (same as before)
    q1_loss = torch.nn.functional.mse_loss(q1_pred, q_target)
    q2_loss = torch.nn.functional.mse_loss(q2_pred, q_target)
    critic_loss = q1_loss + q2_loss

    # Total entropy is sum of continuous and discrete entropies
    total_log_prob = steer_log_prob + up_log_prob + down_log_prob

    # Actor loss
    actor_loss = (alpha * total_log_prob - torch.min(q1_pred, q2_pred)).mean()

    # Temperature loss
    target_entropy = -3.0  # -dim(total action space)
    temp_loss = -(alpha * (total_log_prob + target_entropy).detach()).mean()

    return critic_loss, actor_loss, temp_loss


class Trainer:
    __slots__ = (
        "online_network",
        "target_network",
        "policy_optimizer",
        "q_optimizer",
        "alpha_optimizer",
        "scaler",
        "batch_size",
        "log_alpha",
    )

    def __init__(
        self,
        online_network: SAC_Network,
        target_network: SAC_Network,
        policy_optimizer: torch.optim.Optimizer,
        q_optimizer: torch.optim.Optimizer,
        alpha_optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        batch_size: int,
        log_alpha: torch.Tensor
    ):
        self.online_network = online_network
        self.target_network = target_network
        self.policy_optimizer = policy_optimizer
        self.q_optimizer = q_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.scaler = scaler
        self.batch_size = batch_size
        self.log_alpha = log_alpha

    def train_on_batch(self, buffer: ReplayBuffer, do_learn: bool):
        """
        Implements one iteration of SAC training
        https://spinningup.openai.com/en/latest/algorithms/sac.html?highlight=gammas#pseudocode
        """
        self.policy_optimizer.zero_grad(set_to_none=True)
        self.q_optimizer.zero_grad(set_to_none=True)
        self.alpha_optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            batch, batch_info = buffer.sample(self.batch_size, return_info=True)
            (
                state_img_tensor,
                state_float_tensor,
                actions,
                rewards,
                next_state_img_tensor,
                next_state_float_tensor,
                gammas_terminal,
            ) = batch

            # Get current alpha value
            alpha = self.log_alpha.exp()

            # Compute policy distribution and Q-values for current state
            policy_dist, features = self.online_network(state_img_tensor, state_float_tensor)

            # Sample actions and compute log probs
            sampled_actions = policy_dist.rsample()
            log_prob = policy_dist.log_prob(sampled_actions).sum(dim=-1, keepdim=True)

            # Get Q-values
            q1_pred, q2_pred = self.online_network.get_q_values(features, sampled_actions)

            with torch.no_grad():
                # Compute target Q-values
                next_policy_dist, next_features = self.target_network(next_state_img_tensor, next_state_float_tensor)
                next_actions = next_policy_dist.rsample()
                next_log_prob = next_policy_dist.log_prob(next_actions).sum(dim=-1, keepdim=True)

                next_q1, next_q2 = self.target_network.get_q_values(next_features, next_actions)
                next_q = torch.min(next_q1, next_q2) - alpha * next_log_prob
                q_target = rewards + gammas_terminal * next_q

            # Compute losses
            critic_loss, actor_loss, temp_loss = sac_loss(q1_pred, q2_pred, q_target, log_prob, alpha)
            total_loss = critic_loss + actor_loss + temp_loss

            if do_learn:
                # Single backward pass
                self.scaler.scale(total_loss).backward()

                # Unscale all optimizers
                self.scaler.unscale_(self.q_optimizer)
                self.scaler.unscale_(self.policy_optimizer)
                self.scaler.unscale_(self.alpha_optimizer)

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.online_network.parameters(),
                    config_copy.clip_grad_norm
                ).detach().cpu().item()

                # Step all optimizers
                self.scaler.step(self.q_optimizer)
                self.scaler.step(self.policy_optimizer)
                self.scaler.step(self.alpha_optimizer)

                # Single scaler update
                self.scaler.update()
            else:
                grad_norm = 0

            total_loss = total_loss.detach().cpu()

        return total_loss, grad_norm

class Inferer:
    __slots__ = (
        "inference_network",
        "epsilon",
        "is_explo",
    )

    def __init__(self, inference_network: SAC_Network):
        self.inference_network = inference_network
        self.epsilon = None
        self.is_explo = None

    def infer_network(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray) -> npt.NDArray:
        """
        Perform inference of a single state through self.inference_network.
        """
        with torch.no_grad():
            state_img_tensor = (
               torch.from_numpy(img_inputs_uint8)
               .unsqueeze(0)
               .to("cuda", memory_format=torch.channels_last, non_blocking=True,
                   dtype=torch.float32)
               - 128
            ) / 128
            state_float_tensor = torch.from_numpy(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)
            return self.inference_network(state_img_tensor, state_float_tensor)

    def get_exploration_action(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray) -> Tuple[
        Tuple[float, bool, bool], bool, float, npt.NDArray]:
        """Returns (steer, up, down) action tuple"""
        (steer_dist, up_dist, down_dist), features = self.infer_network(img_inputs_uint8, float_inputs)
        mean, log_std = steer_dist
        steer_dist = torch.distributions.Normal(mean, log_std.exp())
        up_dist = torch.distributions.Bernoulli(logits=up_dist)
        down_dist = torch.distributions.Bernoulli(logits=down_dist)

        r = random.random()

        if self.is_explo and r < self.epsilon:
            # Sample during exploration
            is_greedy = False
            steer = torch.clamp(steer_dist.sample(), -1.0, 1.0).item()
            up = up_dist.sample().bool().item()
            down = down_dist.sample().bool().item()
        else:
            # Use means during exploitation
            is_greedy = True
            steer = steer_dist.mean.item()
            up = (up_dist.probs > 0.5).bool().item()
            down = (down_dist.probs > 0.5).bool().item()

        # Combine actions into a tensor for Q-value computation
        action_tensor = torch.tensor([[steer, float(up), float(down)]], device="cuda")
        q1, q2 = self.inference_network.get_q_values(features, action_tensor)
        value = torch.min(q1, q2).item()

        return (
            (steer, up, down),  # Action tuple
            is_greedy,  # is_greedy
            value,  # Q-value
            action_tensor.cpu().numpy()  # Full action tensor
        )

def make_untrained_sac_network(jit: bool, is_inference: bool) -> Tuple[SAC_Network, SAC_Network]:
    """
    Constructs two identical copies of the SAC network.

    The first copy is compiled (if jit == True) and is used for inference, for rollouts, for training, etc...
    The second copy is never compiled and **only** used to efficiently share neural network weights between processes.

    Args:
        jit: a boolean indicating whether compilation should be used
        is_inference: a boolean indicating whether the model will be used for inference (affects compilation mode)

    Returns:
        Tuple containing:
        - The compiled (if jit=True) model for active use
        - An uncompiled copy for weight sharing
    """
    uncompiled_model = SAC_Network(
        float_inputs_dim=config_copy.float_input_dim,
        float_hidden_dim=config_copy.float_hidden_dim,
        conv_head_output_dim=config_copy.conv_head_output_dim,
        dense_hidden_dimension=config_copy.dense_hidden_dimension,
        float_inputs_mean=config_copy.float_inputs_mean,
        float_inputs_std=config_copy.float_inputs_std,
    )

    if jit:
        if config_copy.is_linux:
            compile_mode = None if "rocm" in torch.__version__ else (
                "max-autotune" if is_inference else "max-autotune-no-cudagraphs")
            model = torch.compile(uncompiled_model, dynamic=False, mode=compile_mode)
        else:
            model = torch.jit.script(uncompiled_model)
    else:
        model = copy.deepcopy(uncompiled_model)

    return (
        model.to(device="cuda", memory_format=torch.channels_last).train(),
        uncompiled_model.to(device="cuda", memory_format=torch.channels_last).train(),
    )