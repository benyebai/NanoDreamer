import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nanodreamer.config import Config
from nanodreamer.losses import lambda_returns
from nanodreamer.models import WorldModel, head


class Agent:
    def __init__(self, channels, actions, device="cpu", config=None):
        self.config = config or Config()
        self.device = torch.device(device)
        settings = self.config

        # instantiating all our networks
        self.world = WorldModel(channels, actions, settings).to(device)
        # this encoder is what encodes our actual next latent vs predicted
        self.target_encoder = copy.deepcopy(self.world.encoder).requires_grad_(False)
        self.actor = head(settings.width, actions, zero=True).to(device)

        # as a note, our critic helps us predict after the 5 step imagined rollout
        # and a critic's purpose is to produces a score:
        #  "this is how much we usually get from this position"
        self.critic = head(settings.width, 1, zero=True).to(device)

        # we introduce a target critic
        # live critic for our baseline (calculating advantage)
        # our target critic is used for lambda return and its updates slowly, slowly towards te live critic
        # so remove gradient, we will handle the update ourselves
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)

        # instantiating all our networks' optimizer
        self.world_opt = torch.optim.Adam(self.world.parameters(), lr=settings.world_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=settings.actor_lr)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=settings.critic_lr
        )

        # in the first draft we just counted it by hand and saw how rare a
        # specific channel was a put a bigger weight, this time we actually move it
        self.density = torch.full((channels,), 0.05, device=device)
        # keep the critic imagined returns in check, or else it can
        # drown out the actors updates
        self.return_scale = 1.0
        self.actions = actions

    def tensors(self, data):
        return {
            name: torch.as_tensor(value, device=self.device)
            for (name, value) in data.items()
        }

    # d = channel density, p = predicted probability the cell is on.
    # On cell:  loss = (0.5 / d) * -log(p)
    # Off cell: loss = (0.5 / (1 - d)) * -log(1 - p)
    # Average over all cells and channels to get one loss per frame.
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

    # update our running history
    def append(self, latents, previous_actions, next_latent, action):
        return (
            torch.cat([latents[:, 1:], next_latent[:, None]], 1),
            torch.cat([previous_actions[:, 1:], action[:, None]], 1),
        )

    # we actually use the state() function that gets us our "summary" latent
    # so we can give it to the actor!
    @torch.no_grad()
    def observe(self, frames, previous_actions):
        context_length = self.config.context
        # create an array of shape (1, context length, 6, 10, 10)
        frame_history = np.zeros((1, context_length, *frames[-1].shape), np.float32)
        # create an array of shape (1, context length, actions)
        action_history = np.full((1, context_length), self.actions, np.int64)
        # might have less than context_lenght amts of frames
        history_length = min(context_length, len(frames))
        frame_history[0, -history_length:] = frames[-history_length:]
        action_history[0, -history_length:] = previous_actions[-history_length:]
        latents = self.world.encoder(torch.as_tensor(frame_history, device=self.device))
        latents[:, : context_length - history_length] = 0
        hidden_state = self.world.states(
            latents, torch.as_tensor(action_history, device=self.device)
        )[:, -1]
        return (latents, hidden_state)

    # taking our hidden state from the function above
    # we then pick an action using our actor!
    # just pick based off of returned probability
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

        # get us random sample batches from our replay, the prioritized gives scoring transitions more chances too
        batch = self.tensors(replay.sample(config.batch, prioritized=True))

        # [:, 1] take last frame of each batch, and get the averages
        # lerp means move it 1% away each time (so slow update)
        self.density.lerp_(batch["frames"][:, -1].mean((0, 2, 3)), 0.01)
        # turn them into our latents
        latents = self.world.encoder(batch["frames"]) * batch["valid"][..., None]
        # get the summary
        hidden_states = self.world.states(latents, batch["previous"])
        # get all their predicted latents
        predicted_latents, _, _ = self.world.predict(hidden_states, batch["actions"])

        # now we use our target encoder to build our answers (encoding the real frames to compare against prediction)
        with torch.no_grad():
            targets = self.target_encoder(batch["next_frames"])
            future_targets = self.target_encoder(batch["future"])

        valid = batch["valid"]

        # recontruction loss
        # our decoded latents vs actual frame
        reconstruction = (
            self.binary_loss(self.world.decoder(latents), batch["frames"]) * valid
        ).sum() / valid.sum()

        # prediction loss = sum(valid × (latent MSE + weighted frame BCE)) / sum(valid)
        # our predicted latents vs target (this is just for next predicted btw)
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
            # take new history summary latent
            state = self.world.states(context, previous)[:, -1]
            # get all our head predicted values
            next_latent, predicted_reward, continuation_logits = self.world.predict(
                state, batch["future_actions"][:, step]
            )

            # now time to cacluate all our losses, MSE, cross entropy, etc
            frame_loss = self.binary_loss(
                self.world.decoder(next_latent), batch["future"][:, step]
            )
            latent_loss = (next_latent - future_targets[:, step]).square().mean(-1)
            reward_error = predicted_reward - batch["rewards"][:, step]
            reward_prediction_loss = reward_error.square()
            continuation_loss = F.binary_cross_entropy_with_logits(
                continuation_logits, batch["continues"][:, step], reduction="none"
            )

            # how important this batch is, like if its a point transition window
            weight = batch["future_valid"][:, step] * batch["importance"]

            # add it up to our running tally
            rollout_loss = rollout_loss + (
                (
                    frame_loss
                    + latent_loss
                    + 2 * reward_prediction_loss
                    + 0.2 * continuation_loss
                )
                * weight
            ).sum() / weight.sum().clamp(min=1)

            # this is just for our recording purposes later
            reward_loss += float(
                (reward_prediction_loss * weight).sum().detach()
                / weight.sum().clamp(min=1)
            )

            # move it over a frame [F1, F2, F3] -> [F2, F3, F4]
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
                # Slowly follow the live encoder: 99% old target + 1% live weights.
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
        # 3 to 5 steps now for imagination
        batch = self.tensors(replay.sample(config.imagination_batch))

        # not updating our world model here
        with torch.no_grad():
            latents = self.world.encoder(batch["frames"]) * batch["valid"][..., None]
        previous = batch["previous"]

        # our ongoing tally now
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

            # pretend actor spits out like: [100%, 0%, 0%, 0%]
            # we turn it into [99.25%, 0.25%, 0.25%, 0.25%]
            probabilities = (
                0.99 * self.actor(hidden_state).softmax(-1) + 0.01 / self.actions
            )
            # just sample our action, take the logprob for later use, and save entropy
            action_distribution = torch.distributions.Categorical(probs=probabilities)
            action = action_distribution.sample()
            log_probabilities.append(action_distribution.log_prob(action))
            # entropy measure how spread, so we want to reward just a tad bit
            # if ur proabbilties are spread evenly!
            entropies.append(action_distribution.entropy())
            states.append(hidden_state)

            # get all our heads
            with torch.no_grad():
                next_latent, predicted_reward, continuation_logits = self.world.predict(
                    hidden_state, action
                )
                rewards.append(predicted_reward)
                continuations.append(continuation_logits.sigmoid())
                latents, previous = self.append(latents, previous, next_latent, action)
                next_states.append(self.world.states(latents, previous)[:, -1])

        states, next_states = (torch.stack(states, 1), torch.stack(next_states, 1))
        rewards, continuations = (
            torch.stack(rewards, 1),
            torch.stack(continuations, 1),
        )

        # actor loss
        with torch.no_grad():
            values = self.target_critic(next_states).squeeze(-1)

            # [G1, G2, G3, G4, G5]
            imagined_returns = lambda_returns(
                rewards, continuations, values, config.discount, config.lam
            )
            baseline = self.critic(states).squeeze(-1)

            # take our 95th and 5th percentle to remove extremes
            return_range = float(
                torch.quantile(imagined_returns.flatten(), 0.95)
                - torch.quantile(imagined_returns.flatten(), 0.05)
            )
            # then move our return scale towards it a bit
            self.return_scale = 0.99 * self.return_scale + 0.01 * max(1.0, return_range)
            # make sure our advantage is within reason
            advantage = (imagined_returns - baseline) / max(1.0, self.return_scale)

            # The first imagined move counts fully because we start there.
            # Weight each later move by the chance of reaching it, with one discount
            # per earlier move. If the episode ends, all moves after it get zero weight.
            weights = torch.cat(
                [
                    torch.ones_like(continuations[:, :1]),
                    (config.discount * continuations[:, :-1]).cumprod(1),
                ],
                1,
            )

        # now we calculate our policy!
        # Make the sampled move more likely if it did better than expected,
        # and less likely if it did worse. Add a small bonus for keeping
        # other moves in play, and give less weight to moves we may never reach.
        # Negate the result because the optimizer minimizes the loss.
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

        # critic loss
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
                next_hidden_state = self.world.states(
                    real_latents, real_previous_actions
                )[:, -1]
                next_values.append(self.target_critic(next_hidden_state).squeeze(-1))
            real_returns = lambda_returns(
                real["rewards"],
                real["continues"],
                torch.stack(next_values, 1),
                config.discount,
                config.lam,
                real["future_valid"],
            )[:, 0]

        # use both iamgined returns and real returns to update critic
        real_error = self.critic(real_hidden_state).squeeze(-1) - real_returns
        imagined_error = self.critic(states).squeeze(-1) - imagined_returns
        real_loss = (real_error.square() * real["importance"]).mean()
        imagined_loss = (imagined_error.square() * weights).mean()

        critic_loss = real_loss + 0.3 * imagined_loss
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_opt.step()

        # update our target critic
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_parameter.lerp_(parameter, 0.02)
        return (float(policy_loss.detach()), float(critic_loss.detach()))
