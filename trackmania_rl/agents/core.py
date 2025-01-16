import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from math import floor


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

def num_flat_features(x):
    size = x.size()[1:]  # dimension 0 is the batch dimension, so it is ignored
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


# The next utility computes the dimensionality of the output in a 2D CNN layer:
def conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)
    return h_out, w_out

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class VanillaCNN(nn.Module):
    def __init__(self, q_net):
        """
        Simple CNN (Convolutional Neural Network) model for SAC (Soft Actor-Critic).

        Args:
            q_net (bool): indicates whether this neural net is a critic network
        """
        super(VanillaCNN, self).__init__()

        self.q_net = q_net

        # Convolutional layers processing screenshots:
        # The default config.json gives 4 grayscale images of 64 x 64 pixels
        self.h_out, self.w_out = config_copy.H_downsized, config_copy.W_downsized
        self.conv1 = nn.Conv2d(1, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels

        # Dimensionality of the CNN output:
        self.flat_features = self.out_channels * self.h_out * self.w_out

        # Dimensionality of the MLP input:
        # The MLP input will be formed of:
        # - the flattened CNN output
        # - the current speed, gear and RPM measurements (3 floats)
        # - the 2 previous actions (2 x 3 floats), important because of the real-time nature of our controller
        # - when the module is the critic, the selected action (3 floats)
        float_features = 12 if self.q_net else 9
        self.mlp_input_features = self.flat_features + float_features

        # MLP layers:
        # (when using the model as a policy, we will sample from a multivariate gaussian defined later in the tutorial;
        # thus, the output dimensionality is  1 for the critic, and we will define the output layer of policies later)
        self.mlp_layers = [256, 256, 1] if self.q_net else [256, 256]
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.LeakyReLU)

    def forward(self, x):
        """
        In Pytorch, the forward function is where our neural network computes its output from its input.

        Args:
            x (torch.Tensor): input tensor (i.e., the observation fed to our deep neural network)

        Returns:
            the output of our neural network in the form of a torch.Tensor
        """
        if self.q_net:
            # The critic takes the next action (act) as additional input
            # act1 and act2 are the actions in the action buffer (real-time RL):
            speed, gear, rpm, images, act1, act2, act = x
        else:
            # For the policy, the next action (act) is what we are computing, so we don't have it:
            speed, gear, rpm, images, act1, act2 = x

        # Forward pass of our images in the CNN:
        # (note that the competition environment outputs histories of 4 images
        # and the default config outputs these as 64 x 64 greyscale,
        # we will stack these greyscale images along the channel dimension of our input tensor)
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Now we will flatten our output feature map.
        # Let us double-check that our dimensions are what we expect them to be:
        flat_features = num_flat_features(x)
        assert flat_features == self.flat_features, f"x.shape:{x.shape},\
                                                    flat_features:{flat_features},\
                                                    self.out_channels:{self.out_channels},\
                                                    self.h_out:{self.h_out},\
                                                    self.w_out:{self.w_out}"
        # All good, let us flatten our output feature map:
        x = x.view(-1, flat_features)

        # Finally, we can feed the result along our float values to the MLP:
        if self.q_net:
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)
        else:
            x = torch.cat((speed, gear, rpm, x, act1, act2), -1)
        x = self.mlp(x)

        # And this gives us the output of our deep neural network :)
        return x

class FeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        float_inputs_dim,
        float_hidden_dim,
        float_inputs_mean,
        float_inputs_std,
    ):
        super().__init__()
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
            torch.nn.Linear(float_inputs_dim, float_hidden_dim),
            activation_function(inplace=True),
            torch.nn.Linear(float_hidden_dim, float_hidden_dim),
            activation_function(inplace=True),
        )
        self.float_inputs_mean = torch.tensor(float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(float_inputs_std, dtype=torch.float32).to("cuda")

    def initialize_weights(self):
        lrelu_neg_slope = 1e-2
        lrelu_gain = torch.nn.init.calculate_gain("leaky_relu", lrelu_neg_slope)
        for m in self.img_head:
            if isinstance(m, torch.nn.Conv2d):
                nn_utilities.init_orthogonal(m, lrelu_gain)
        for m in self.float_feature_extractor:
            if isinstance(m, torch.nn.Linear):
                nn_utilities.init_orthogonal(m, lrelu_gain)

    def forward(self, img, float_inputs, use_fp32: bool = False) -> torch.Tensor:
        img_outputs = self.img_head((img.to(torch.float32 if use_fp32 else torch.float16) - 128) / 128)  # PERF
        # img_outputs = torch.zeros(batch_size, misc.conv_head_output_dim).to(device="cuda") # Uncomment to temporarily mask the img_head
        float_outputs = self.float_feature_extractor((float_inputs - self.float_inputs_mean) / self.float_inputs_std)
        # (batch_size, dense_input_dimension) OK
        return torch.cat((img_outputs, float_outputs), 1)


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
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
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.__fe = FeatureExtractor(obs_dim, act_dim, hidden_sizes, activation)
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()