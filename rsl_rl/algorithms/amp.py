# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.modules import ActorCritic
from rsl_rl.modules import Discriminator
from rsl_rl.amp_motion_loader import AMPTocabiMotionLoader
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage, ReplayBuffer
from rsl_rl.utils import string_to_callable


class AMP:
    """AMP algorithm (https://arxiv.org/abs/2104.02180)."""

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # LCP parameters
        lcp_cfg: dict | None = None,
        # Bound loss parameters
        bound_loss_cfg: dict | None = None,
        # AMP parameters
        amp_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Extract learning rate and remove it from the original dict
            learning_rate = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=learning_rate)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        if lcp_cfg is not None:
            self.use_lcp = lcp_cfg
        else:
            self.use_lcp = None

        if bound_loss_cfg is not None:
            self.use_bound_loss = bound_loss_cfg
        else:
            self.use_bound_loss = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)

        # AMP components
        self.amp_cfg = amp_cfg
        self.disc_cfg = self.amp_cfg["disc_cfg"]
        assert self.amp_cfg is not None, "AMP configuration is not provided."
        self.discriminator = Discriminator(
            num_amp_obs=self.amp_cfg["num_amp_obs_per_step"] * self.amp_cfg["num_amp_obs_steps"],
            **self.disc_cfg
        ).to(self.device)
        self.amp_storage = ReplayBuffer(self.discriminator.num_amp_obs, self.amp_cfg["amp_replay_buffer_size"], self.device)
        self.amp_transition = RolloutStorage.Transition()
        self.amp_data = AMPTocabiMotionLoader(self.amp_cfg["motion_file"], self.amp_cfg["num_amp_obs_steps"], self.amp_cfg["env_dt"], self.device)

        # Create optimizer
        params = [
            {"params": self.policy.parameters(), "name": "policy"},
            {"params": self.discriminator.parameters(), "name": "disc"},
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)

        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs, amp_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        self.amp_transition.observations = amp_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the AMP observations
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_disc_loss = 0
        mean_disc_pen_loss = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None
        # -- LCP loss
        if self.use_lcp:
            mean_gradient_penalty_loss = 0
        else:
            mean_gradient_penalty_loss = None
        # -- Bound loss
        if self.use_bound_loss:
            mean_bound_loss = 0
        else:
            mean_bound_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # AMP generator
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)

        # iterate over batches

        for (ppo_batch, amp_policy_batch, amp_expert_batch) in zip(generator, amp_policy_generator, amp_expert_generator):
            # unpack the PPO batch
            (   obs_batch, 
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                rnd_state_batch) = ppo_batch
            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            if self.use_lcp:
                print(f"obs_batch.requires_grad: {obs_batch.requires_grad}")
                obs_est_batch = obs_batch.clone()
                obs_est_batch.requires_grad_()
                print(f"obs_batch.requires_grad: {obs_batch.requires_grad}")
                print(f"obs_est_batch.requires_grad: {obs_est_batch.requires_grad}")
                self.policy.act(obs_est_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            else:
                self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])

            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.use_lcp:
                gradient_penalty_loss = self._calc_grad_penalty(obs_est_batch, actions_log_prob_batch)

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Discriminator loss
            policy_amp_obs = amp_policy_batch
            expert_amp_obs = amp_expert_batch

            policy_disc_output = self.discriminator.discriminate(policy_amp_obs)
            expert_disc_output = self.discriminator.discriminate(expert_amp_obs)

            assert self.amp_cfg is not None, "AMP configuration is not provided."
            if self.disc_cfg["gan_type"] == "vanilla":
                policy_loss = torch.nn.BCEWithLogitsLoss()(policy_disc_output, torch.zeros(policy_disc_output.size(), device=self.device))
                expert_loss = torch.nn.BCEWithLogitsLoss()(expert_disc_output, torch.ones(expert_disc_output.size(), device=self.device))
                grad_pen_loss = self.discriminator.compute_grad_pen(amp_expert_batch, lambda_=self.amp_cfg["disc_grad_pen"])
                disc_loss = (expert_loss + policy_loss) *0.5
            elif self.disc_cfg["gan_type"] == "lsgan":
                # LSGAN
                policy_loss = torch.nn.MSELoss()(
                    policy_disc_output, -1 * torch.ones(policy_disc_output.size(), device=self.device))
                expert_loss = torch.nn.MSELoss()(
                    expert_disc_output, torch.ones(expert_disc_output.size(), device=self.device))
                grad_pen_loss = self.discriminator.compute_grad_pen(amp_expert_batch, lambda_=self.amp_cfg["disc_grad_pen"])
                disc_loss = (expert_loss + policy_loss) *0.5
            elif self.disc_cfg["gan_type"] == "wgan":
                # WGAN
                alpha = 0.5 # 0.1 ~ 0.5 is recommended
                policy_loss = torch.nn.Tanh()(policy_disc_output * alpha).mean()
                expert_loss = -torch.nn.Tanh()(expert_disc_output * alpha).mean()
                grad_pen_loss = self.discriminator.compute_grad_pen(amp_expert_batch, lambda_=self.amp_cfg["disc_grad_pen"])
                disc_loss = (expert_loss + policy_loss) *0.5
            else:
                assert False, f"Invalid GAN type: {self.disc_cfg['gan_type']}"

            disc_logit_loss = torch.sum(torch.square(self.discriminator.get_logit_weight()))

            loss = surrogate_loss \
                + self.value_loss_coef * value_loss \
                - self.entropy_coef * entropy_batch.mean() \
                + self.amp_cfg["disc_loss_coef"] * disc_loss \
                + self.amp_cfg["disc_logit_loss_coef"] * disc_logit_loss \
                + grad_pen_loss
            
            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            if self.use_lcp:
                gradient_penalty_coef = self.use_lcp["gradient_penalty_coef"]
                loss += gradient_penalty_coef * gradient_penalty_loss

            if self.use_bound_loss:
                bound_loss_coef = self.use_bound_loss["bound_loss_coef"]
                bound_loss = self.bound_loss(mu_batch)
                loss += bound_loss_coef * bound_loss 

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            print(f"backward done")
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_disc_loss += disc_loss.item()
            # mean_disc_pen_loss += grad_pen_loss.item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # -- LCP loss
            if mean_gradient_penalty_loss is not None:
                mean_gradient_penalty_loss += gradient_penalty_loss.item()
            # -- Bound loss
            if mean_bound_loss is not None:
                mean_bound_loss += bound_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For Discriminator loss
        mean_disc_loss /= num_updates
        mean_disc_pen_loss /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- For LCP
        if mean_gradient_penalty_loss is not None:
            mean_gradient_penalty_loss /= num_updates
        # -- For Bound loss
        if mean_bound_loss is not None:
            mean_bound_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "disc": mean_disc_loss,
            "disc_pen": mean_disc_pen_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.use_lcp:
            loss_dict["lcp"] = mean_gradient_penalty_loss
        if self.use_bound_loss:
            loss_dict["bound"] = mean_bound_loss

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel

    def _calc_grad_penalty(self, obs_batch, actions_log_prob_batch):
        grad_log_prob = torch.autograd.grad(actions_log_prob_batch.sum(), obs_batch, create_graph=True, retain_graph=True)[0]
        gradient_penalty_loss = torch.sum(torch.square(grad_log_prob), dim=-1).mean()
        return gradient_penalty_loss
    
    def bound_loss(self, mu):
        soft_bound = self.use_bound_loss["bound_range"]
        mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
        mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
        b_loss = (mu_loss_low + mu_loss_high).sum()
        return b_loss