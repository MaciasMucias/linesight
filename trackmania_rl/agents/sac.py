import copy
import sys

import torch
import numpy as np
import numpy.typing as npt
import torch.nn.functional as F
from typing import Tuple
import math
from torch.optim import Optimizer
from trackmania_rl.buffer_management import ReplayBuffer
from config_files import config_copy


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

        self.LOG_STD_MIN = -5
        self.LOG_STD_MAX = 2
        self.EPS = 1e-6
        self.MIN_Q_VALUE = -50.0
        self.MAX_Q_VALUE = 50.0

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
        self.steer_mean = torch.nn.Linear(dense_hidden_dimension, 1)
        self.steer_log_std = torch.nn.Linear(dense_hidden_dimension, 1)
        self.up_logits = torch.nn.Linear(dense_hidden_dimension, 1)
        self.down_logits = torch.nn.Linear(dense_hidden_dimension, 1)

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
        for module in [self.steer_mean, self.steer_log_std, self.up_logits, self.down_logits]:
            torch.nn.init.uniform_(module.weight, -3e-3, 3e-3)
            torch.nn.init.uniform_(module.bias, -3e-3, 3e-3)

            # Special initialization for final Q-network layers
        for q_net in [self.q1_net, self.q2_net]:
            final_layer = q_net[-1]
            torch.nn.init.uniform_(final_layer.weight, -3e-3, 3e-3)
            torch.nn.init.uniform_(final_layer.bias, -3e-3, 3e-3)

    def preprocess_inputs(self, img: torch.Tensor, float_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess inputs consistently, whether coming from inferer or learner

        Args:
            img: Image tensor, either:
                - Already normalized [-1, 1] from learner
                - Raw [0, 255] from inferer
            float_inputs: Float tensor

        Returns:
            Tuple of (processed_img, processed_float)
        """
        # Check if image needs normalization (from inferer)
        if img.dtype in (torch.uint8, torch.int32, torch.int64):
            img = ((img.float() - 128) / 128)

        # Normalize float inputs
        processed_float = (float_inputs - self.float_inputs_mean) / self.float_inputs_std

        return img, processed_float

    def get_features(self, img: torch.Tensor, float_inputs: torch.Tensor) -> torch.Tensor:
        """Get features using preprocessed inputs"""
        img, float_inputs = self.preprocess_inputs(img, float_inputs)

        img_outputs = self.img_head(img)
        float_outputs = self.float_feature_extractor(float_inputs)
        features = torch.cat((img_outputs, float_outputs), 1)
        return self.shared_net(features)

    def sample_normal(self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool) -> Tuple[
        torch.Tensor, torch.Tensor]:
        if deterministic:
            return mean, torch.zeros_like(mean)

        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        sample = mean + eps * std
        log_prob = (-0.5 * ((eps ** 2) + 2.0 * log_std + math.log(2 * math.pi)))
        return sample, log_prob

    def forward(self, img: torch.Tensor, float_inputs: torch.Tensor,
                deterministic: bool, with_logprob: bool) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.get_features(img, float_inputs)

        # Get steering distribution parameters
        mean = self.steer_mean(features)
        mean = torch.clamp(mean, -3.0, 3.0)
        log_std = torch.clamp(self.steer_log_std(features), self.LOG_STD_MIN, self.LOG_STD_MAX)

        # Sample steering action
        if deterministic:
            steer_action = torch.tanh(mean)
            log_prob_steer = torch.zeros_like(mean)
        else:
            std = torch.exp(log_std)
            noise = torch.randn_like(mean)
            raw_steer_action = mean + noise * std
            steer_action = torch.tanh(raw_steer_action)

            # More numerically stable log prob computation
            log_prob_steer = (-0.5 * (noise.pow(2) + 2 * log_std + math.log(2 * math.pi)) -
                        torch.log(1 - steer_action.pow(2) + self.EPS))

        # Get gas and brake probabilities
        gas_logits = self.up_logits(features)
        brake_logits = self.down_logits(features)
        gas_logits = torch.clamp(gas_logits, -10.0, 10.0)
        brake_logits = torch.clamp(brake_logits, -10.0, 10.0)

        gas_prob = torch.sigmoid(gas_logits)
        brake_prob = torch.sigmoid(brake_logits)

        # Sample discrete actions
        if deterministic:
            gas_action = (gas_prob > 0.5).float()
            brake_action = (brake_prob > 0.5).float()
        else:
            uniform_noise = torch.rand_like(gas_prob)
            gas_action = (uniform_noise < gas_prob).float()
            brake_action = (uniform_noise < brake_prob).float()

        # Compute log probabilities
        log_prob_gas = -F.binary_cross_entropy_with_logits(gas_logits, gas_action, reduction='none')
        log_prob_brake = -F.binary_cross_entropy_with_logits(brake_logits, brake_action, reduction='none')

        log_prob = (
                self.steer_entropy_scale * log_prob_steer +
                self.discrete_entropy_scale * (log_prob_gas + log_prob_brake)
        )

        if with_logprob:
            log_prob = log_prob + 0.1 * (log_prob_gas + log_prob_brake)
        else:
            log_prob = torch.zeros_like(log_prob)

        return steer_action, gas_action, brake_action, log_prob

    @torch.jit.export
    def q_values(self, img, float_inputs, steer, gas, brake):
        features = self.get_features(img, float_inputs)
        # Create new tensors instead of inplace operations
        actions = torch.cat([
            steer.detach().clone(),
            gas.detach().clone(),
            brake.detach().clone()
        ], dim=-1)
        sa = torch.cat([features.detach().clone(), actions], dim=-1)

        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)

        # Create new tensors instead of inplace clamping
        q1 = torch.max(torch.min(q1, torch.tensor(self.MAX_Q_VALUE, device=q1.device)),
                       torch.tensor(self.MIN_Q_VALUE, device=q1.device))
        q2 = torch.max(torch.min(q2, torch.tensor(self.MAX_Q_VALUE, device=q2.device)),
                       torch.tensor(self.MIN_Q_VALUE, device=q2.device))

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
        actions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # steer, gas, brake
        rewards: torch.Tensor,
        next_state_img: torch.Tensor,
        next_state_float: torch.Tensor,
        gammas: torch.Tensor,
        log_alpha: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alpha = log_alpha.exp()

    # Get current Q values
    current_q1, current_q2 = online_network.q_values(
        state_img, state_float, *actions
    )

    # Sample actions and compute Q-values for next state
    with torch.no_grad():
        next_steer, next_gas, next_brake, next_log_prob = target_network(
            next_state_img, next_state_float, deterministic=False, with_logprob=True
        )

        # Target Q-values
        target_q1, target_q2 = target_network.q_values(
            next_state_img, next_state_float,
            next_steer, next_gas, next_brake
        )
        target_q = torch.min(target_q1, target_q2)

        # Explicitly reshape next_log_prob
        next_log_prob = next_log_prob.view(-1, 1)  # [batch_size, 1]

        # Compute target value with entropy - note the sign change for entropy term
        target_value = target_q - alpha * next_log_prob

        # Ensure rewards and gammas are correct shape
        rewards = rewards.view(-1, 1)  # [batch_size, 1]
        gammas = gammas.view(-1, 1)  # [batch_size, 1]

        # Calculate target
        q_target = rewards + gammas * target_value
        q_target = q_target.view(-1, 1).detach()  # Force correct shape

    # Q-function loss (critic loss) - MSE between current and target Q-values
    q1_loss = F.mse_loss(current_q1, q_target, reduction='mean')
    q2_loss = F.mse_loss(current_q2, q_target, reduction='mean')
    critic_loss = q1_loss + q2_loss

    # Policy loss (actor loss) - sample new actions for the policy gradient
    steer, gas, brake, log_prob = online_network(
        state_img, state_float, deterministic=False, with_logprob=True
    )
    q1_pi, q2_pi = online_network.q_values(state_img, state_float, steer, gas, brake)
    min_q_pi = torch.min(q1_pi, q2_pi)

    log_prob = log_prob.view(-1, 1)
    # Corrected policy loss: maximize Q-value and entropy

    policy_loss = (-min_q_pi - alpha * log_prob).mean()

    # Temperature loss - automatic entropy tuning
    target_entropy = -3.0  # Can be tuned based on action space
    alpha_loss = -(log_alpha * (log_prob.detach() + target_entropy)).mean()

    return critic_loss, policy_loss, alpha_loss


class Trainer:
    def __init__(
            self,
            online_network: torch.nn.Module,
            target_network: torch.nn.Module,
            policy_optimizer: Optimizer,
            q_optimizer: Optimizer,
            alpha_optimizer: Optimizer,
            scaler: torch.amp.GradScaler,
            batch_size: int,
            log_alpha: torch.Tensor,
    ):
        self.online_network = online_network
        self.target_network = target_network
        self.policy_optimizer = policy_optimizer
        self.q_optimizer = q_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.scaler = scaler
        self.batch_size = batch_size
        self.log_alpha = log_alpha
        self.train_steps = 0

        # Compute target entropy based on action space
        # -1 for steering (continuous) and -0.98 for each discrete action
        self.target_entropy = -(1 + 2 * 0.98)

    def train_on_batch(self, buffer: ReplayBuffer, do_learn: bool):
        try:
            critic_grad_norm = 0.0
            policy_grad_norm = 0.0

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                # Get batch
                batch, batch_info = buffer.sample(self.batch_size, return_info=True)
                (state_img_tensor, state_float_tensor, actions, rewards,
                 next_state_img_tensor, next_state_float_tensor, gammas) = batch

                # === Critic Update ===
                if do_learn:
                    self.q_optimizer.zero_grad(set_to_none=True)

                    with torch.set_grad_enabled(True):
                        # Fresh forward pass for critic
                        current_q1, current_q2 = self.online_network.q_values(
                            state_img_tensor,
                            state_float_tensor,
                            actions[:, 0].unsqueeze(-1),
                            actions[:, 1].unsqueeze(-1),
                            actions[:, 2].unsqueeze(-1)
                        )

                        with torch.no_grad():
                            # Compute targets
                            next_actions = self.target_network(
                                next_state_img_tensor,
                                next_state_float_tensor,
                                deterministic=False,
                                with_logprob=True
                            )
                            next_steer, next_gas, next_brake, next_log_prob = next_actions

                            target_q1, target_q2 = self.target_network.q_values(
                                next_state_img_tensor,
                                next_state_float_tensor,
                                next_steer,
                                next_gas,
                                next_brake
                            )

                            alpha = self.log_alpha.exp()
                            min_target_q = torch.min(target_q1, target_q2)
                            target_value = min_target_q - alpha * next_log_prob.view(-1, 1)
                            q_target = rewards.view(-1, 1) + gammas.view(-1, 1) * target_value

                        # Compute critic loss
                        critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)

                        # Update critic
                        self.scaler.scale(critic_loss).backward()
                        self.scaler.unscale_(self.q_optimizer)
                        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                            [p for name, p in self.online_network.named_parameters()
                             if any(x in name for x in ['q1_net', 'q2_net'])],
                            float('inf')
                        ).item()
                        self.scaler.step(self.q_optimizer)
                        self.scaler.update()

                # === Policy Update ===
                if do_learn:
                    self.policy_optimizer.zero_grad(set_to_none=True)

                    with torch.set_grad_enabled(True):
                        # Fresh forward pass for policy
                        pi_steer, pi_gas, pi_brake, log_prob = self.online_network(
                            state_img_tensor,
                            state_float_tensor,
                            deterministic=False,
                            with_logprob=True
                        )

                        q1_pi, q2_pi = self.online_network.q_values(
                            state_img_tensor,
                            state_float_tensor,
                            pi_steer,
                            pi_gas,
                            pi_brake
                        )

                        min_q_pi = torch.min(q1_pi, q2_pi)
                        alpha = self.log_alpha.exp()

                        # Compute policy loss
                        policy_loss = (alpha * log_prob - min_q_pi).mean()

                        # Update policy
                        self.scaler.scale(policy_loss).backward()
                        self.scaler.unscale_(self.policy_optimizer)
                        policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                            [p for name, p in self.online_network.named_parameters()
                             if any(x in name for x in ['steer_mean', 'steer_log_std', 'up_logits', 'down_logits'])],
                            float('inf')
                        ).item()
                        self.scaler.step(self.policy_optimizer)
                        self.scaler.update()

                    # === Alpha Update ===
                    self.alpha_optimizer.zero_grad(set_to_none=True)
                    alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
                    self.scaler.scale(alpha_loss).backward()
                    self.scaler.unscale_(self.alpha_optimizer)
                    self.scaler.step(self.alpha_optimizer)
                    self.scaler.update()

                    # === Target Network Update ===
                    with torch.no_grad():
                        for target_param, online_param in zip(
                                self.target_network.parameters(),
                                self.online_network.parameters()
                        ):
                            target_param.data.copy_(
                                0.995 * target_param.data + 0.005 * online_param.data
                            )

                # Return values even when not learning
                return (
                    float(critic_loss + policy_loss + alpha_loss),
                    float(critic_loss),
                    float(policy_loss),
                    float(alpha_loss),
                    float(-log_prob.mean()),
                    max(critic_grad_norm, policy_grad_norm)
                )

        except Exception as e:
            print(f"Unexpected error in train_on_batch: {str(e)}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


class Inferer:
    """Handles inference for SAC policy network"""
    __slots__ = ("inference_network", "is_explo")

    def __init__(self, inference_network: torch.nn.Module):
        self.inference_network = inference_network
        self.is_explo = None

    def _prepare_tensors(self, img_inputs_uint8: np.ndarray, float_inputs: np.ndarray) -> tuple[
        torch.Tensor, torch.Tensor]:
        """Prepare input tensors for network inference"""
        state_img_tensor = (
            torch.from_numpy(img_inputs_uint8)
            .unsqueeze(0)
            .to("cuda", memory_format=torch.channels_last, non_blocking=True)
        )

        state_float_tensor = torch.from_numpy(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)
        return state_img_tensor, state_float_tensor

    def infer_network(self, img_inputs_uint8: np.ndarray, float_inputs: np.ndarray) -> np.ndarray:
        """Get Q-values for a given state using deterministic policy"""
        with torch.no_grad():
            state_img_tensor, state_float_tensor = self._prepare_tensors(img_inputs_uint8, float_inputs)

            steer, gas, brake, _ = self.inference_network(
                state_img_tensor,
                state_float_tensor,
                deterministic=True,
                with_logprob=False
            )

            q1, q2 = self.inference_network.q_values(
                state_img_tensor,
                state_float_tensor,
                steer, gas, brake
            )

            return torch.min(q1, q2).cpu().numpy()

    def get_exploration_action(
            self,
            img_inputs_uint8: np.ndarray,
            float_inputs: np.ndarray
    ) -> tuple[np.ndarray, bool, float, np.ndarray]:
        """Select an action using the policy, with optional exploration during training"""
        with torch.no_grad():
            state_img_tensor, state_float_tensor = self._prepare_tensors(img_inputs_uint8, float_inputs)

            # Get deterministic action for value estimation
            steer_det, gas_det, brake_det, _ = self.inference_network(
                state_img_tensor,
                state_float_tensor,
                deterministic=True,
                with_logprob=False
            )

            q1, q2 = self.inference_network.q_values(
                state_img_tensor,
                state_float_tensor,
                steer_det, gas_det, brake_det
            )
            value = torch.min(q1, q2).item()

            # During exploration, sample from policy
            if self.is_explo:
                steer, gas, brake, _ = self.inference_network(
                    state_img_tensor,
                    state_float_tensor,
                    deterministic=False,
                    with_logprob=False
                )
            else:
                steer, gas, brake = steer_det, gas_det, brake_det

            action = np.array([
                steer.cpu().numpy()[0, 0],
                gas.cpu().numpy()[0, 0],
                brake.cpu().numpy()[0, 0]
            ])

            deterministic_action = np.array([
                steer_det.cpu().numpy()[0, 0],
                gas_det.cpu().numpy()[0, 0],
                brake_det.cpu().numpy()[0, 0]
            ])

            return action, np.allclose(action, deterministic_action), value, q1.cpu().numpy()


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
        n_actions=len(["steer", "gas", "break"]),
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