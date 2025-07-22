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
        activation="elu",
        normalize_amp_obs=True,
        **kwargs,
    ):
        if kwargs:
            print(
                "Discriminator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

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

        print(f"Discriminator MLP: {self.disc}")


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
