import math
from numbers import Number

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyexpat import features

from torch.distributions import constraints, Distribution
from torch.distributions.normal import Normal
from math import floor
from config_files import config_copy
from config_files.config import float_hidden_dim


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


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])



class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.float_inputs_dim = config_copy.float_input_dim
        self.float_hidden_dim = config_copy.float_hidden_dim

        img_head_channels = [1, 16, 32, 64, 32]
        activation_function = torch.nn.LeakyReLU
        self.img_head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=img_head_channels[0], out_channels=img_head_channels[1], kernel_size=(4, 4), stride=2),
            activation_function(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[1], out_channels=img_head_channels[2], kernel_size=(4, 4), stride=2),
            activation_function(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[2], out_channels=img_head_channels[3], kernel_size=(3, 3), stride=2),
            activation_function(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[3], out_channels=img_head_channels[4], kernel_size=(3, 3), stride=1),
            activation_function(inplace=True),
            torch.nn.Flatten(),
        )
        self.float_feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(self.float_inputs_dim, self.float_hidden_dim),
            activation_function(inplace=True),
            torch.nn.Linear(self.float_hidden_dim, self.float_hidden_dim),
            activation_function(inplace=True),
        )
        self.float_inputs_mean = torch.tensor(config_copy.float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(config_copy.float_inputs_std, dtype=torch.float32).to("cuda")

        self.output_dim = config_copy.conv_head_output_dim + self.float_hidden_dim

    def forward(self, obs: tuple[torch.Tensor, torch.Tensor], use_fp32: bool = True) -> torch.Tensor:
        img, float_inputs = obs
        img_outputs = self.img_head((img.to(torch.float32 if use_fp32 else torch.float16) - 128) / 128)  # PERF
        # img_outputs = torch.zeros(batch_size, misc.conv_head_output_dim).to(device="cuda") # Uncomment to temporarily mask the img_head
        float_outputs = self.float_feature_extractor((float_inputs - self.float_inputs_mean) / self.float_inputs_std)
        # (batch_size, dense_input_dimension) OK
        return torch.cat((img_outputs, float_outputs), 1)


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self,
                 hidden_sizes,
                 act_dim,
                 act_limit,
                 activation
                 ):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.net = mlp([self.feature_extractor.output_dim] + list(hidden_sizes), activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, obs: tuple[torch.Tensor, torch.Tensor], deterministic: bool=False, with_logprob: bool=True):
        features = self.feature_extractor(obs)
        net_out = self.net(features)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(dim=-1)
            logp_pi -= (2*(torch.log(torch.tensor(2)) - pi_action - F.softplus(-2*pi_action))).sum(dim=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, hidden_sizes, activation):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.q = mlp([self.feature_extractor.output_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs: tuple[torch.Tensor, torch.Tensor], act):
        features = self.feature_extractor(obs)
        q = self.q(torch.cat([features, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

