import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from config_files import config_copy


class Normal:
    """Jit compilation workaround"""
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

    def rsample(self) -> torch.Tensor:
        epsilon = torch.randn_like(self.scale)
        return self.loc + self.scale * epsilon

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        var = self.variance
        log_scale = torch.log(self.scale)
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

def mlp(sizes, activation, output_activation=nn.Identity):
    """Create a MLP with specified sizes and activation"""
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.float_inputs_dim = config_copy.float_input_dim
        self.float_hidden_dim = config_copy.float_hidden_dim

        img_head_channels = [1, 16, 32, 64, 32]
        activation_function = torch.nn.LeakyReLU

        # CNN for image input
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

        # MLP for floats input
        self.float_feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(self.float_inputs_dim, self.float_hidden_dim),
            activation_function(inplace=True),
            torch.nn.Linear(self.float_hidden_dim, self.float_hidden_dim),
            activation_function(inplace=True),
        )

        # Register buffers for float input normalization
        self.register_buffer('float_inputs_mean',
                             torch.tensor(config_copy.float_inputs_mean, dtype=torch.float32))
        self.register_buffer('float_inputs_std',
                             torch.tensor(config_copy.float_inputs_std, dtype=torch.float32))

        self.output_dim = config_copy.conv_head_output_dim + self.float_hidden_dim

    def forward(self, obs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Image input normalization
        img, float_inputs = obs
        img = (img - 128.0) / 128.0
        img_outputs = self.img_head(img)

        # Float input normalization
        float_outputs = self.float_feature_extractor(
            (float_inputs - self.float_inputs_mean) / self.float_inputs_std
        )

        return torch.cat((img_outputs, float_outputs), 1)

class Actor(nn.Module):
    def __init__(self, act_dim, act_limit, hidden_sizes, activation):
        super().__init__()

        self.feature_extractor = FeatureExtractor()
        self.net = mlp([self.feature_extractor.output_dim] + list(hidden_sizes), activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

        # Constants corresponding to STD_MAX = 7.3 and STD_MIN = 2e-9. This shouldn't ever be an issue
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

        # Register log(2) as a buffer for TorchScript compatibility
        self.register_buffer('log_2', torch.log(torch.tensor(2.0, dtype=torch.float32)))

    def forward(self, obs: tuple[torch.Tensor, torch.Tensor], deterministic: bool = False, with_logprob: bool = True):
        features = self.feature_extractor(obs)
        net_out = self.net(features)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        # Calculate the logprob
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(dim=-1)
            logp_pi -= (2 * (self.log_2 - pi_action - F.softplus(-2 * pi_action))).sum(dim=1)
        else:
            logp_pi = None


        # This ensures actions are in range [-act_limit, act_limit].
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class QFunction(nn.Module):
    def __init__(self, act_dim, hidden_sizes, activation):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.q = mlp([self.feature_extractor.output_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs: tuple[torch.Tensor, torch.Tensor], act):
        features = self.feature_extractor(obs)
        fa = torch.cat([features, act], dim=-1)
        q = self.q(fa)
        return torch.squeeze(q, -1)

