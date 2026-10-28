"""Episode replay with optional reward-event sampling."""

import numpy as np


class Replay:
    def __init__(self, config, channels, actions, seed):
        self.config, self.channels, self.actions = (config, channels, actions)
        self.episodes = []
        self.rng = np.random.default_rng(seed)
        self.total = 0
        self.events = []
        self.event_offset = 0

    def start(self, observation):
        self.episodes.append(
            dict(frames=[observation.copy()], actions=[], rewards=[], continues=[])
        )

    def add(self, action, reward, continuation, observation):
        episode = self.episodes[-1]
        episode["actions"].append(action)
        episode["rewards"].append(reward)
        episode["continues"].append(continuation)
        episode["frames"].append(observation.copy())
        self.total += 1
        if reward != 0:
            self.events.append((episode, len(episode["actions"]) - 1))
        while len(self.episodes) > 1 and self.total > self.config.replay_capacity:
            old_episode = self.episodes.pop(0)
            self.total -= len(old_episode["actions"])
            while (
                self.event_offset < len(self.events)
                and self.events[self.event_offset][0] is old_episode
            ):
                self.event_offset += 1
        if self.event_offset > 10000:
            self.events = self.events[self.event_offset :]
            self.event_offset = 0

    def sample(self, batch, prioritized=False):
        lengths = np.array([len(episode["actions"]) for episode in self.episodes])
        cumulative = np.cumsum(lengths)
        total = int(cumulative[-1])
        event_count = len(self.events) - self.event_offset if prioritized else 0
        sampled_starts, weights = ([], [])
        event_fraction = 0.25 if event_count else 0.0
        for _ in range(batch):
            if self.rng.random() < event_fraction:
                episode, start_step = self.events[
                    self.event_offset + int(self.rng.integers(event_count))
                ]
            else:
                transition_index = int(self.rng.integers(total))
                episode_index = int(
                    np.searchsorted(cumulative, transition_index, side="right")
                )
                start_step = transition_index - (
                    int(cumulative[episode_index - 1]) if episode_index else 0
                )
                episode = self.episodes[episode_index]
            sampled_starts.append((episode, start_step))
            # Correct for sampling reward events more often.
            event_boost = (
                event_fraction * total / event_count
                if episode["rewards"][start_step] != 0 and event_count
                else 0
            )
            weights.append(1 / ((1 - event_fraction) + event_boost))
        context_length, horizon = (self.config.context, self.config.world_horizon)
        frames = np.zeros((batch, context_length, self.channels, 10, 10), np.float32)
        next_frames = frames.copy()
        previous = np.full((batch, context_length), self.actions, np.int64)
        actions = previous.copy()
        valid = np.zeros((batch, context_length), np.float32)
        future = np.zeros((batch, horizon, self.channels, 10, 10), np.float32)
        future_actions = np.zeros((batch, horizon), np.int64)
        rewards = np.zeros((batch, horizon), np.float32)
        continues = rewards.copy()
        future_valid = rewards.copy()
        for row, (episode, start_step) in enumerate(sampled_starts):
            first_step = max(0, start_step - context_length + 1)
            history_length = start_step - first_step + 1
            frames[row, -history_length:] = episode["frames"][
                first_step : start_step + 1
            ]
            next_frames[row, -history_length:] = episode["frames"][
                first_step + 1 : start_step + 2
            ]
            previous[row, -history_length:] = [
                self.actions if step == 0 else episode["actions"][step - 1]
                for step in range(first_step, start_step + 1)
            ]
            actions[row, -history_length:] = episode["actions"][
                first_step : start_step + 1
            ]
            valid[row, -history_length:] = 1
            for future_step in range(horizon):
                if start_step + future_step < len(episode["actions"]):
                    future[row, future_step] = episode["frames"][
                        start_step + future_step + 1
                    ]
                    future_actions[row, future_step] = episode["actions"][
                        start_step + future_step
                    ]
                    rewards[row, future_step] = episode["rewards"][
                        start_step + future_step
                    ]
                    continues[row, future_step] = episode["continues"][
                        start_step + future_step
                    ]
                    future_valid[row, future_step] = 1
        return dict(
            frames=frames,
            next_frames=next_frames,
            previous=previous,
            actions=actions,
            valid=valid,
            future=future,
            future_actions=future_actions,
            rewards=rewards,
            continues=continues,
            future_valid=future_valid,
            importance=np.array(weights, np.float32),
        )
