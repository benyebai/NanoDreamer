import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nanodreamer.config import Config
from nanodreamer.losses import distribution_loss, distribution_mean, lambda_returns
from nanodreamer.models import WorldModel, head


class Agent:
    def __init__(self, channels, actions, device="cpu", config=None):
        self.config = config or Config()
        self.device = torch.device(device)
        settings = self.config
        self.world = WorldModel(channels, actions, settings).to(device)
        self.target_encoder = copy.deepcopy(self.world.encoder).requires_grad_(False)
        self.actor = head(settings.width, actions, zero=True).to(device)
        self.critic = head(settings.width, 255, zero=True).to(device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.world_opt = torch.optim.Adam(self.world.parameters(), lr=settings.world_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=settings.actor_lr)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=settings.critic_lr
        )
        self.density = torch.full((channels,), 0.05, device=device)
        self.return_scale = 1.0
        self.actions = actions

    def tensors(self, data):
        return {
            name: torch.as_tensor(value, device=self.device)
            for (name, value) in data.items()
        }

    def binary_loss(self, logits, targets):
        channel_density = self.density.clamp(0.001, 0.999)
        weight = torch.where(
            targets.bool(),
            0.5 / channel_density[:, None, None],
            0.5 / (1 - channel_density[:, None, None]),
        )
        return (
            F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            * weight
        ).mean((-3, -2, -1))

    def append(self, latents, previous_actions, next_latent, action):
        return (
            torch.cat([latents[:, 1:], next_latent[:, None]], 1),
            torch.cat([previous_actions[:, 1:], action[:, None]], 1),
        )

    @torch.no_grad()
    def observe(self, frames, previous_actions):
        context_length = self.config.context
        frame_history = np.zeros((1, context_length, *frames[-1].shape), np.float32)
        action_history = np.full((1, context_length), self.actions, np.int64)
        history_length = min(context_length, len(frames))
        frame_history[0, -history_length:] = frames[-history_length:]
        action_history[0, -history_length:] = previous_actions[-history_length:]
        latents = self.world.encoder(torch.as_tensor(frame_history, device=self.device))
        latents[:, : context_length - history_length] = 0
        hidden_state = self.world.states(
            latents, torch.as_tensor(action_history, device=self.device)
        )[:, -1]
        return (latents, hidden_state)

    @torch.no_grad()
    def action(self, frames, previous_actions, rng, epsilon=0.0):
        _, hidden_state = self.observe(frames, previous_actions)
        logits = self.actor(hidden_state)
        probabilities = 0.99 * logits.softmax(-1) + 0.01 / self.actions
        choice = (
            int(rng.integers(self.actions))
            if rng.random() < epsilon
            else int(torch.multinomial(probabilities[0], 1).item())
        )
        return choice

    def update(self, replay):
        config = self.config
        batch = self.tensors(replay.sample(config.batch, prioritized=True))
        self.density.lerp_(batch["frames"][:, -1].mean((0, 2, 3)), 0.01)
        latents = self.world.encoder(batch["frames"]) * batch["valid"][..., None]
        hidden_states = self.world.states(latents, batch["previous"])
        predicted_latents, _, _ = self.world.predict(hidden_states, batch["actions"])
        with torch.no_grad():
            targets = self.target_encoder(batch["next_frames"])
            future_targets = self.target_encoder(batch["future"])
        valid = batch["valid"]
        reconstruction = (
            self.binary_loss(self.world.decoder(latents), batch["frames"]) * valid
        ).sum() / valid.sum()
        prediction = (
            (
                (predicted_latents - targets).square().mean(-1)
                + self.binary_loss(
                    self.world.decoder(predicted_latents), batch["next_frames"]
                )
            )
            * valid
        ).sum() / valid.sum()
        context, previous = (latents, batch["previous"])
        rollout_loss = torch.zeros((), device=self.device)
        reward_loss = 0.0
        for step in range(config.world_horizon):
            state = self.world.states(context, previous)[:, -1]
            next_latent, reward_logits, continuation_logits = self.world.predict(
                state, batch["future_actions"][:, step]
            )
            frame_loss = self.binary_loss(
                self.world.decoder(next_latent), batch["future"][:, step]
            )
            latent_loss = (next_latent - future_targets[:, step]).square().mean(-1)
            reward_prediction_loss = distribution_loss(
                reward_logits, batch["rewards"][:, step]
            )
            continuation_loss = F.binary_cross_entropy_with_logits(
                continuation_logits, batch["continues"][:, step], reduction="none"
            )
            weight = batch["future_valid"][:, step] * batch["importance"]
            rollout_loss = rollout_loss + (
                (
                    frame_loss
                    + latent_loss
                    + 2 * reward_prediction_loss
                    + 0.2 * continuation_loss
                )
                * weight
            ).sum() / weight.sum().clamp(min=1)
            reward_loss += float(
                (reward_prediction_loss * weight).sum().detach()
                / weight.sum().clamp(min=1)
            )
            context, previous = self.append(
                context, previous, next_latent, batch["future_actions"][:, step]
            )
        loss = reconstruction + prediction + rollout_loss / config.world_horizon
        self.world_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.world.parameters(), 5.0)
        self.world_opt.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_encoder.parameters(), self.world.encoder.parameters()
            ):
                target_parameter.lerp_(parameter, 0.01)
        self.world_opt.zero_grad(set_to_none=True)
        actor_loss, critic_loss = self.update_behavior(replay, batch)
        return dict(
            world_loss=float(loss.detach()),
            reconstruction=float(reconstruction.detach()),
            reward_loss=reward_loss / config.world_horizon,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
        )

    def update_behavior(self, replay, real):
        config = self.config
        batch = self.tensors(replay.sample(config.imagination_batch))
        with torch.no_grad():
            latents = self.world.encoder(batch["frames"]) * batch["valid"][..., None]
        previous = batch["previous"]
        states, next_states, rewards, continuations, log_probabilities, entropies = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for _ in range(config.imagination_horizon):
            with torch.no_grad():
                hidden_state = self.world.states(latents, previous)[:, -1]
            probabilities = (
                0.99 * self.actor(hidden_state).softmax(-1) + 0.01 / self.actions
            )
            action_distribution = torch.distributions.Categorical(probs=probabilities)
            action = action_distribution.sample()
            log_probabilities.append(action_distribution.log_prob(action))
            entropies.append(action_distribution.entropy())
            states.append(hidden_state)
            with torch.no_grad():
                next_latent, reward_logits, continuation_logits = self.world.predict(
                    hidden_state, action
                )
                rewards.append(distribution_mean(reward_logits))
                continuations.append(continuation_logits.sigmoid())
                latents, previous = self.append(latents, previous, next_latent, action)
                next_states.append(self.world.states(latents, previous)[:, -1])
        states, next_states = (torch.stack(states, 1), torch.stack(next_states, 1))
        rewards, continuations = (
            torch.stack(rewards, 1),
            torch.stack(continuations, 1),
        )
        with torch.no_grad():
            values = distribution_mean(self.target_critic(next_states))
            imagined_returns = lambda_returns(
                rewards, continuations, values, config.discount, config.lam
            )
            baseline = distribution_mean(self.critic(states))
            return_range = float(
                torch.quantile(imagined_returns.flatten(), 0.95)
                - torch.quantile(imagined_returns.flatten(), 0.05)
            )
            self.return_scale = 0.99 * self.return_scale + 0.01 * max(1.0, return_range)
            advantage = (imagined_returns - baseline) / max(1.0, self.return_scale)
            weights = torch.cat(
                [
                    torch.ones_like(continuations[:, :1]),
                    (config.discount * continuations[:, :-1]).cumprod(1),
                ],
                1,
            )
        policy_loss = -(
            weights
            * (
                torch.stack(log_probabilities, 1) * advantage
                + config.entropy * torch.stack(entropies, 1)
            )
        ).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()
        # Train the critic on real returns as well as imagined returns.
        with torch.no_grad():
            real_latents = self.world.encoder(real["frames"]) * real["valid"][..., None]
            real_previous_actions = real["previous"]
            real_hidden_state = self.world.states(real_latents, real_previous_actions)[
                :, -1
            ]
            next_values = []
            for step in range(config.world_horizon):
                next_latent = self.world.encoder(real["future"][:, step])
                real_latents, real_previous_actions = self.append(
                    real_latents,
                    real_previous_actions,
                    next_latent,
                    real["future_actions"][:, step],
                )
                next_values.append(
                    distribution_mean(
                        self.target_critic(
                            self.world.states(real_latents, real_previous_actions)[
                                :, -1
                            ]
                        )
                    )
                )
            real_returns = lambda_returns(
                real["rewards"],
                real["continues"],
                torch.stack(next_values, 1),
                config.discount,
                config.lam,
                real["future_valid"],
            )[:, 0]
        critic_loss = (
            distribution_loss(self.critic(real_hidden_state), real_returns)
            * real["importance"]
        ).mean()
        critic_loss = (
            critic_loss
            + 0.3
            * (
                distribution_loss(self.critic(states), imagined_returns) * weights
            ).mean()
        )
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_opt.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_parameter.lerp_(parameter, 0.02)
        return (float(policy_loss.detach()), float(critic_loss.detach()))
