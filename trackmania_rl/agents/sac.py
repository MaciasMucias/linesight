import copy
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

        # Image processing head
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

        # Float feature processing
        self.float_feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(float_inputs_dim, float_hidden_dim),
            activation_function(inplace=True),
            torch.nn.Linear(float_hidden_dim, float_hidden_dim),
            activation_function(inplace=True),
        )

        # Combined features dimension
        self.dense_input_dimension = conv_head_output_dim + float_hidden_dim

        # Shared feature network
        self.shared_net = torch.nn.Sequential(
            torch.nn.Linear(self.dense_input_dimension, dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, dense_hidden_dimension),
            activation_function(inplace=True),
        )

        # Policy heads
        self.steer_mean = torch.nn.Linear(dense_hidden_dimension, 1)
        self.steer_log_std = torch.nn.Linear(dense_hidden_dimension, 1)
        self.up_logits = torch.nn.Linear(dense_hidden_dimension, 1)
        self.down_logits = torch.nn.Linear(dense_hidden_dimension, 1)

        # Q-value networks
        self.q1_net = torch.nn.Sequential(
            torch.nn.Linear(dense_hidden_dimension + 3, dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, 1)
        )

        self.q2_net = torch.nn.Sequential(
            torch.nn.Linear(dense_hidden_dimension + 3, dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, dense_hidden_dimension),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, 1)
        )

        self.float_inputs_mean = torch.tensor(float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(float_inputs_std, dtype=torch.float32).to("cuda")

        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Initialize convolutional layers
        for m in self.img_head.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)

        # Initialize float feature extractor
        for m in self.float_feature_extractor.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)

        # Initialize shared network
        for m in self.shared_net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)

        # Initialize policy heads with smaller gain
        for module in [self.steer_mean, self.steer_log_std, self.up_logits, self.down_logits]:
            torch.nn.init.orthogonal_(module.weight, gain=0.01)
            torch.nn.init.zeros_(module.bias)

        # Initialize Q networks
        for qnet in [self.q1_net, self.q2_net]:
            for m in qnet.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    torch.nn.init.zeros_(m.bias)

    def get_features(self, img: torch.Tensor, float_inputs: torch.Tensor) -> torch.Tensor:
        img_outputs = self.img_head(img)
        float_outputs = self.float_feature_extractor(
            (float_inputs - self.float_inputs_mean) / self.float_inputs_std
        )
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
        log_std = torch.clamp(self.steer_log_std(features), -20, 2)

        # Sample steering action
        raw_steer_action, raw_log_prob = self.sample_normal(mean, log_std, deterministic)
        steer_action = torch.tanh(raw_steer_action)

        # Get gas and brake probabilities
        gas_logits = self.up_logits(features)
        brake_logits = self.down_logits(features)

        gas_prob = torch.sigmoid(gas_logits)
        brake_prob = torch.sigmoid(brake_logits)

        # Sample discrete actions
        if deterministic:
            gas_action = (gas_prob > 0.5).float()
            brake_action = (brake_prob > 0.5).float()
        else:
            gas_action = (torch.rand_like(gas_prob) < gas_prob).float()
            brake_action = (torch.rand_like(brake_prob) < brake_prob).float()

        # Compute log probabilities
        log_prob_steer = raw_log_prob - torch.log(1 - steer_action.pow(2) + 1e-6)
        log_prob_gas = -F.binary_cross_entropy_with_logits(gas_logits, gas_action, reduction='none')
        log_prob_brake = -F.binary_cross_entropy_with_logits(brake_logits, brake_action, reduction='none')

        log_prob = log_prob_steer + log_prob_gas + log_prob_brake

        if not with_logprob:
            log_prob = torch.zeros_like(log_prob)

        return steer_action, gas_action, brake_action, log_prob

    @torch.jit.export
    def q_values(self, img: torch.Tensor, float_inputs: torch.Tensor,
                 steer: torch.Tensor, gas: torch.Tensor, brake: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.get_features(img, float_inputs)
        # Concatenate features with actions
        actions = torch.cat([steer, gas, brake], dim=-1)
        sa = torch.cat([features, actions], dim=-1)
        return self.q1_net(sa), self.q2_net(sa)

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
    """
    Compute SAC losses for critic, actor and temperature
    """
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

        # Compute target value with entropy
        target_value = target_q - alpha * next_log_prob
        q_target = rewards + gammas * target_value

    # Q-function loss (critic loss)
    q1_loss = F.mse_loss(current_q1, q_target)
    q2_loss = F.mse_loss(current_q2, q_target)
    critic_loss = q1_loss + q2_loss

    # Policy loss (actor loss)
    steer, gas, brake, log_prob = online_network(
        state_img, state_float, deterministic=False, with_logprob=True
    )
    q1_pi, q2_pi = online_network.q_values(state_img, state_float, steer, gas, brake)
    min_q_pi = torch.min(q1_pi, q2_pi)

    policy_loss = (alpha * log_prob - min_q_pi).mean()

    # Temperature loss
    target_entropy = -3.0  # Heuristic value: -dim(A) for continuous actions
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

    def train_on_batch(self, buffer: ReplayBuffer, do_learn: bool) -> Tuple[float, float]:
        """
        Implements one iteration of SAC training
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
                gammas,
            ) = batch

            # Split the actions into steer, gas, brake
            steer = actions[:, 0].unsqueeze(-1)
            gas = actions[:, 1].unsqueeze(-1)
            brake = actions[:, 2].unsqueeze(-1)

            if config_copy.prio_alpha > 0:
                IS_weights = torch.from_numpy(batch_info["_weight"]).to("cuda", non_blocking=True)

            critic_loss, policy_loss, alpha_loss = sac_loss(
                self.online_network,
                self.target_network,
                state_img_tensor,
                state_float_tensor,
                (steer, gas, brake),
                rewards,
                next_state_img_tensor,
                next_state_float_tensor,
                gammas,
                self.log_alpha
            )

            # Apply importance sampling weights if using prioritized replay
            if config_copy.prio_alpha > 0:
                critic_loss = (critic_loss * IS_weights).mean()
                policy_loss = (policy_loss * IS_weights).mean()
                alpha_loss = (alpha_loss * IS_weights).mean()

            total_loss = critic_loss + policy_loss + alpha_loss

            if do_learn:
                # Update critics
                self.scaler.scale(critic_loss).backward(retain_graph=True)

                # Update policy
                self.scaler.scale(policy_loss).backward(retain_graph=True)

                # Update temperature
                self.scaler.scale(alpha_loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.policy_optimizer)
                self.scaler.unscale_(self.q_optimizer)
                self.scaler.unscale_(self.alpha_optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.online_network.parameters(),
                    config_copy.clip_grad_norm
                ).detach().cpu().item()

                torch.nn.utils.clip_grad_value_(
                    self.online_network.parameters(),
                    config_copy.clip_grad_value
                )

                # Optimizer steps
                self.scaler.step(self.policy_optimizer)
                self.scaler.step(self.q_optimizer)
                self.scaler.step(self.alpha_optimizer)
                self.scaler.update()
            else:
                grad_norm = 0

        return total_loss.detach().cpu(), grad_norm


class Inferer:
    def __init__(self, inference_network):
        self.inference_network = inference_network
        self.epsilon = None  # For compatibility with exploration logic
        self.epsilon_boltzmann = None
        self.is_explo = None

    def infer_network(self, img_inputs_uint8: np.ndarray, float_inputs: np.ndarray) -> np.ndarray:
        """
        Perform inference on a single state
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

            # Get actions from policy network
            steer, gas, brake, _ = self.inference_network(
                state_img_tensor,
                state_float_tensor,
                deterministic=True,
                with_logprob=False
            )

            # Get Q-values for these actions
            q1, q2 = self.inference_network.q_values(
                state_img_tensor,
                state_float_tensor,
                steer, gas, brake
            )

            q_values = torch.min(q1, q2).cpu().numpy()

            return q_values

    def get_exploration_action(
            self,
            img_inputs_uint8: np.ndarray,
            float_inputs: np.ndarray
    ) -> Tuple[np.ndarray, bool, float, np.ndarray]:
        """
        Select an action using the policy, with potential exploration
        Returns: action, is_greedy, value, q_values
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

            # Get deterministic action (for value estimation)
            steer_det, gas_det, brake_det, _ = self.inference_network(
                state_img_tensor,
                state_float_tensor,
                deterministic=True,
                with_logprob=False
            )

            # Get Q-values for deterministic action
            q1, q2 = self.inference_network.q_values(
                state_img_tensor,
                state_float_tensor,
                steer_det, gas_det, brake_det
            )
            value = torch.min(q1, q2).item()

            # During exploration, sample from the policy
            if self.is_explo:
                steer, gas, brake, _ = self.inference_network(
                    state_img_tensor,
                    state_float_tensor,
                    deterministic=False,
                    with_logprob=False
                )
            else:
                steer, gas, brake = steer_det, gas_det, brake_det

            # Compose action array
            action = np.array([
                steer.cpu().numpy()[0, 0],
                gas.cpu().numpy()[0, 0],
                brake.cpu().numpy()[0, 0]
            ])

            # Check if this was a greedy action
            is_greedy = np.allclose(
                action,
                np.array([
                    steer_det.cpu().numpy()[0, 0],
                    gas_det.cpu().numpy()[0, 0],
                    brake_det.cpu().numpy()[0, 0]
                ])
            )

            return action, is_greedy, value, q1.cpu().numpy()


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
        n_actions=len(config_copy.inputs),
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