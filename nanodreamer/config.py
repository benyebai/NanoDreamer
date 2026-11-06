"""Settings for the final recipe."""

from dataclasses import dataclass


@dataclass
class Config:
    latent: int = 64
    width: int = 128
    layers: int = 2
    context: int = 16
    batch: int = 32
    imagination_batch: int = 64
    # How many steps the world model predicts forward during its own training.
    world_horizon: int = 3
    # How many steps the actor practises inside the world model. The target critic estimates what’s left beyond those five steps
    imagination_horizon: int = 5
    train_every: int = 8
    seed_steps: int = 4000
    replay_capacity: int = 200000
    world_lr: float = 0.0002
    actor_lr: float = 4e-05
    critic_lr: float = 0.0001
    discount: float = 0.99
    lam: float = 0.95
    # this is for our actor to pick different options
    entropy: float = 0.001
