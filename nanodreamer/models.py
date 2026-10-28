"""CNN encoder, decoder and causal transformer."""

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, channels, latent):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, 3),
            nn.ELU(),
            nn.Conv2d(16, 32, 3),
            nn.ELU(),
            nn.Conv2d(32, 64, 3),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent),
            nn.LayerNorm(latent),
            nn.Tanh(),
        )

    def forward(self, frames):
        batch_shape = frames.shape[:-3]
        return self.cnn(frames.reshape(-1, *frames.shape[-3:])).reshape(
            *batch_shape, -1
        )


class Decoder(nn.Module):
    def __init__(self, channels, latent):
        super().__init__()
        self.linear = nn.Linear(latent, 64 * 4 * 4)
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3),
            nn.ELU(),
            nn.ConvTranspose2d(32, 16, 3),
            nn.ELU(),
            nn.ConvTranspose2d(16, channels, 3),
        )
        self.channels = channels

    def forward(self, latents):
        batch_shape = latents.shape[:-1]
        return self.cnn(self.linear(latents).reshape(-1, 64, 4, 4)).reshape(
            *batch_shape, self.channels, 10, 10
        )


class WorldModel(nn.Module):
    def __init__(self, channels, actions, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(channels, config.latent)
        self.decoder = Decoder(channels, config.latent)
        self.actions = actions
        self.action_embedding = nn.Embedding(actions + 1, 16)
        self.input = nn.Linear(config.latent + 16, config.width)
        self.position = nn.Parameter(torch.zeros(config.context, config.width))
        layer = nn.TransformerEncoderLayer(
            config.width,
            4,
            4 * config.width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, config.layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.width)
        self.transition = nn.Sequential(
            nn.Linear(config.width + 16, config.width), nn.ELU()
        )
        self.latent_head = nn.Linear(config.width, config.latent)
        self.reward_head = nn.Linear(config.width, 255)
        self.continue_head = nn.Linear(config.width, 1)
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)

    def states(self, latents, previous_actions):
        """Read past frames and actions without looking ahead."""
        context_length = latents.shape[1]
        tokens = self.input(
            torch.cat([latents, self.action_embedding(previous_actions)], -1)
        )
        tokens = tokens + self.position[-context_length:]
        mask = torch.ones(
            (context_length, context_length), dtype=torch.bool, device=latents.device
        ).triu(1)
        return self.norm(self.transformer(tokens, mask=mask))

    def predict(self, hidden_state, actions):
        features = self.transition(
            torch.cat([hidden_state, self.action_embedding(actions)], -1)
        )
        return (
            self.latent_head(features).tanh(),
            self.reward_head(features),
            self.continue_head(features).squeeze(-1),
        )


def head(width, output, zero=False):
    model = nn.Sequential(
        nn.Linear(width, width),
        nn.ELU(),
        nn.Linear(width, width),
        nn.ELU(),
        nn.Linear(width, output),
    )
    if zero:
        nn.init.zeros_(model[-1].weight)
        nn.init.zeros_(model[-1].bias)
    return model
