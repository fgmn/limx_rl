# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

from .mlp_encoder import MLP_Encoder
from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage


class PPO:
    actor_critic: ActorCritic
    teacher_encoder: MLP_Encoder
    student_encoder: MLP_Encoder

    def __init__(
        self,
        num_group,
        teacher_encoder,
        student_encoder,
        actor_critic,
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
        vae_beta=1.0,
        student_learning_rate=5.0e-4,
        early_stop=False,
        anneal_lr=False,
        teacher_mode=True,
        device="cpu",
    ):
        self.device = device
        self.num_group = num_group

        self.desired_kl = desired_kl
        self.early_stop = early_stop
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.vae_beta = vae_beta

        self.teacher_encoder = teacher_encoder
        self.teacher_encoder.to(self.device)
        self.student_encoder = student_encoder
        self.student_encoder.to(self.device)

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam([{"params": self.actor_critic.parameters()}], lr=learning_rate)

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

        # Teacher mode 
        if not teacher_mode:
            import copy
            actor = copy.deepcopy(self.actor_critic.actor.eval())
            if self.device is not None:
                actor.to(self.device)
            self.teacher_inference_actor = actor.eval()

            self.student_optimizer = optim.Adam([{"params": self.student_encoder.parameters()}], lr=student_learning_rate)

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_history_shape,
        commands_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_history_shape,
            commands_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, obs_history, commands, critic_obs, teacher_mode=True):
        critic_obs = torch.cat((critic_obs, commands), dim=-1)
        # act
        if teacher_mode:
            teacher_encoder_out = self.teacher_encoder.encode(torch.cat((critic_obs, obs_history), dim=-1))
            self.transition.actions = self.actor_critic.act(
                torch.cat((critic_obs[:, :3], teacher_encoder_out, obs, commands), dim=-1)
            ).detach()
        else:
            student_encoder_out = self.student_encoder.encode(obs_history)
            self.transition.actions = self.actor_critic.act(
                torch.cat((student_encoder_out, obs, commands), dim=-1)
            ).detach()

        # evaluate
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()

        # storage
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_obs = critic_obs
        self.transition.observation_history = obs_history
        self.transition.commands = commands
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.transition.next_observations = next_obs
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, teacher_mode=True, student_finetune_mode=False):
        if teacher_mode and student_finetune_mode:
            raise ValueError("teacher_mode and student_finetune_mode cannot be both True")
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0
        mean_vel_loss = 0
        mean_student_encoder_loss = 0
        mean_student_action_loss = 0
        generator = self.storage.mini_batch_generator(
            self.num_group,
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for (
            obs_batch,
            critic_obs_batch,
            obs_history_batch, _,
            group_commands_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:

            commands_batch = group_commands_batch

            if teacher_mode:
                teacher_encoder_out_batch = self.teacher_encoder.encode(torch.cat((critic_obs_batch, obs_history_batch), dim=-1))
                encoder_out_batch = torch.cat((critic_obs_batch[:, :3], teacher_encoder_out_batch), dim=-1)
            else:
                student_encoder_out_batch = self.student_encoder.encode(obs_history_batch)
                student_action_batch = self.actor_critic.act( torch.cat((student_encoder_out_batch.detach(), obs_batch, commands_batch), dim=-1) )

                teacher_encoder_out_target_batch = self.teacher_encoder.encode(torch.cat((critic_obs_batch, obs_history_batch), dim=-1)).detach()
                if not student_finetune_mode:
                    vel_teacher_encoder_out_target_batch = torch.cat((critic_obs_batch[:, :3], teacher_encoder_out_target_batch), dim=-1)
                    teacher_action_target_batch = self.teacher_inference_actor( torch.cat(
                                                                                (vel_teacher_encoder_out_target_batch, obs_batch, commands_batch),
                                                                                dim=-1,
                                                                                )
                                                                              )
                else:
                    encoder_out_batch = student_encoder_out_batch.detach()


            if teacher_mode or student_finetune_mode:  # Teacher PPO Training or Student PPO Finetuning
                self.actor_critic.act(
                    torch.cat(
                        (encoder_out_batch, obs_batch, commands_batch),
                        dim=-1,
                    )
                )

                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                    actions_batch
                )

                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                kl_mean = torch.tensor(0, device=self.device, requires_grad=False)
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                # KL
                if self.desired_kl != None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                if self.desired_kl != None and self.early_stop:
                    if kl_mean > self.desired_kl * 1.5:
                        print("early stop, num_updates =", num_updates)
                        break

                # Surrogate loss
                ratio = torch.exp(
                    actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
                )

                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (
                        value_batch - target_values_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                entropy_batch_mean = entropy_batch.mean()
                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch_mean
                )

                if self.anneal_lr:
                    frac = 1.0 - num_updates / (
                        self.num_learning_epochs * self.num_mini_batches
                    )
                    self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                num_updates += 1
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_kl += kl_mean.item()

            if not teacher_mode and not student_finetune_mode:  # Student Distillation
                # vel loss
                vel_loss = 5 * (student_encoder_out_batch[:, :1] - critic_obs_batch[:, :1]).pow(2).mean()
                vel_loss += (student_encoder_out_batch[:, 1:3] - critic_obs_batch[:, 1:3]).pow(2).mean()

                # Student encoder loss
                student_encoder_loss = (student_encoder_out_batch[:, 3:] - teacher_encoder_out_target_batch.detach()).pow(2).mean()

                # Student action loss
                student_action_loss = (student_action_batch - teacher_action_target_batch.detach()).pow(2).mean()

                # Combined student loss
                student_loss = student_encoder_loss + student_action_loss + vel_loss

                # Optimize step
                self.student_optimizer.zero_grad()
                student_loss.backward()
                nn.utils.clip_grad_norm_(list(self.student_encoder.parameters()) + list(self.actor_critic.actor.parameters()), self.max_grad_norm)
                self.student_optimizer.step()
                num_updates += 1
                mean_vel_loss += vel_loss.item()
                mean_student_encoder_loss += student_encoder_loss.item()
                mean_student_action_loss += student_action_loss.item()

            if not teacher_mode and student_finetune_mode:  # Only Student Encoder Distillation
                # vel loss
                vel_loss = 5 * (student_encoder_out_batch[:, :1] - critic_obs_batch[:, :1]).pow(2).mean()
                vel_loss += (student_encoder_out_batch[:, 1:3] - critic_obs_batch[:, 1:3]).pow(2).mean()

                # Student encoder loss
                student_encoder_loss = (student_encoder_out_batch[:, 3:] - teacher_encoder_out_target_batch.detach()).pow(2).mean()

                # Combined student loss
                student_loss = student_encoder_loss + vel_loss

                # Optimize step
                self.student_optimizer.zero_grad()
                student_loss.backward()
                nn.utils.clip_grad_norm_(list(self.student_encoder.parameters()) + list(self.actor_critic.actor.parameters()), self.max_grad_norm)
                self.student_optimizer.step()
                num_updates += 1
                mean_vel_loss += vel_loss.item()
                mean_student_encoder_loss += student_encoder_loss.item()

        self.storage.clear()

        if teacher_mode:  # Teacher PPO Training
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            mean_kl /= num_updates
            return mean_value_loss, mean_surrogate_loss, mean_kl
        elif student_finetune_mode:  # Student PPO Finetuning
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            mean_kl /= num_updates
            mean_vel_loss /= num_updates
            mean_student_encoder_loss /= num_updates
            return mean_value_loss, mean_surrogate_loss, mean_kl, mean_vel_loss, mean_student_encoder_loss
        else:  # Student Distillation
            mean_vel_loss /= num_updates
            mean_student_encoder_loss /= num_updates
            mean_student_action_loss /= num_updates
            return mean_vel_loss, mean_student_encoder_loss, mean_student_action_loss

