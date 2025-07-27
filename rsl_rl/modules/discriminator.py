# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules import EmpiricalNormalization

class Discriminator(nn.Module):

    def __init__(
        self,
        num_amp_obs,
        num_amp_output,
        hidden_dims=[256, 256],
        activation="relu",
        normalize_amp_obs=True,
        amp_reward_weight=1.0,
        gan_type="vanilla",
        **kwargs,
    ):
        if kwargs:
            print(
                "Discriminator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.num_amp_obs = num_amp_obs
        self.amp_reward_weight = amp_reward_weight
        self.gan_type = gan_type

        activation = resolve_nn_activation(activation)

        # Discriminator network
        disc_layers = []
        disc_layers.append(nn.Linear(num_amp_obs, hidden_dims[0]))
        disc_layers.append(activation)
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                disc_layers.append(nn.Linear(hidden_dims[layer_index], num_amp_output))
            else:
                disc_layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
                disc_layers.append(activation)
        self.disc = nn.Sequential(*disc_layers)

        if normalize_amp_obs:
            self.amp_normalizer = EmpiricalNormalization(shape=num_amp_obs, until=1.0e8)
        else:
            self.amp_normalizer = None

        print(f"Discriminator MLP with {self.gan_type} type: {self.disc}")


    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def discriminate(self, observations, **kwargs):
        if self.amp_normalizer is not None:
            observations = self.amp_normalizer(observations)
        return self.disc(observations)
    
    def calc_amp_reward(self, observations):
        with torch.no_grad():
            self.eval()
            disc_output = self.discriminate(observations)
            if self.gan_type == "vanilla":
                prob = torch.sigmoid(disc_output)
                amp_reward = torch.log(torch.maximum(1 - prob, torch.tensor(1e-8)))
            elif self.gan_type == "lsgan":
                amp_reward = torch.maximum(1.0 - 0.25 * torch.square(disc_output - 1.0), torch.tensor(0.0))
            elif self.gan_type == "wgan":
                amp_reward = torch.exp(disc_output)
            else:
                assert False, f"Invalid GAN type: {self.gan_type}"
            self.train()
            return self.amp_reward_weight * amp_reward.squeeze()

    def compute_grad_pen(self, demo_amp_obs, lambda_=10):
        demo_amp_obs_est = demo_amp_obs.detach().clone()

        # disc = self.discriminate(demo_amp_obs_est)
        demo_amp_obs_est = self.amp_normalizer(demo_amp_obs_est)
        demo_amp_obs_est.requires_grad_()
        disc = self.disc(demo_amp_obs_est)
        ones = torch.ones(disc.size(), device=disc.device)
        grad = torch.autograd.grad( # Computes the gradients of outputs w.r.t. inputs.
            outputs=disc, inputs=demo_amp_obs_est,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0] # the index [0] indicates gradient w.r.t. the first input. For multiple inputs, use inputs=[input1, input2]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        # grad_pen = lambda_ * torch.sum(torch.square(grad), dim=-1).mean()
        return grad_pen
    
    def compute_grad_pen_interpolate(self, expert_amp_obs, policy_amp_obs, lambda_=10):
        interpolate_data = slerp(expert_amp_obs, policy_amp_obs, torch.rand((expert_amp_obs.shape[0], 1)))

        disc = self.discriminate(interpolate_data)
        ones = torch.ones(disc.size(), device=disc.device)
        grad = torch.autograd.grad( # Computes the gradients of outputs w.r.t. inputs.
            outputs=disc, inputs=interpolate_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0] # the index [0] indicates gradient w.r.t. the first input. For multiple inputs, use inputs=[input1, input2]

        # Enforce that the grad norm approaches 0.
        # grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        grad_pen = lambda_ * (torch.clamp(grad.norm(2, dim=1) - 1., min=0.)).pow(2).mean()
        return grad_pen
    
    def get_logit_weight(self):
        return torch.flatten(self.disc[-1].weight)
    
    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
    
@staticmethod
def slerp(a, b, blend):
    return (1-blend)*a + blend*b