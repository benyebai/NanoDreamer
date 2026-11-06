"""CNN encoder, decoder and causal transformer."""

import torch
from torch import nn


# this is our CNN encoder, basically smaller grids move over the whole input, then spit out a feature map
# i.e: finding a projectile, finding enemies, etc
# then we get muiltiple summaries and try to reason about them by combining them to a latent
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
        # (32, 16, 6, 10, 10) -> (512, 6, 10, 10) [512 frames, each frame 10x10 with 6 channels]
        batch_shape = frames.shape[:-3]
        return self.cnn(frames.reshape(-1, *frames.shape[-3:])).reshape(
            *batch_shape, -1
        )


# basically doing the reverse of the encoder, taking number and turning them into 3x3 patches
# has its own learned weights but trains together with encoder
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
        # holds the latent size, width, block, etc
        self.config = config

        # our autoencoder
        self.encoder = Encoder(channels, config.latent)
        self.decoder = Decoder(channels, config.latent)

        self.actions = actions

        # create the action embeddings with +1 for do nothing, 16 features
        self.action_embedding = nn.Embedding(actions + 1, 16)
        # add frame features with action features -> width of transformer
        self.input = nn.Linear(config.latent + 16, config.width)
        # matrix of 16 rows, each rows 128 features (this is for our 16-frame-history positions)
        self.position = nn.Parameter(torch.zeros(config.context, config.width))

        layer = nn.TransformerEncoderLayer(
            # size of our latents 128
            config.width,
            # heads
            4,
            # size of our feedforward
            4 * config.width,
            dropout=0.0,
            activation="gelu",
            # batch size as our first dimension
            batch_first=True,
            # layernorm before our feedfoward
            norm_first=True,
        )
        # our full transformer
        self.transformer = nn.TransformerEncoder(layer, config.layers)

        # normalize the results after the transformer
        self.norm = nn.LayerNorm(config.width)

        self.transition = nn.Sequential(
            nn.Linear(config.width + 16, config.width), nn.ELU()
        )

        # Predict one raw reward per action, as in the first draft.
        self.latent_head = nn.Linear(config.width, config.latent)
        self.reward_head = nn.Linear(config.width, 1)
        self.continue_head = nn.Linear(config.width, 1)

        # Start with a reward prediction of zero.
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)

    def states(self, latents, previous_actions):
        # latents = (32, 16, 64) 32 batches of 16 frames each
        context_length = latents.shape[1]
        # add the action embeddings
        tokens = self.input(
            torch.cat([latents, self.action_embedding(previous_actions)], -1)
        )
        # add the positional embeddings
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
            # the tanh for the latent head keeps it within the encoders range
            self.latent_head(features).tanh(),
            self.reward_head(features).squeeze(-1),
            self.continue_head(features).squeeze(-1),
        )


# small networks that we will use as our actor and critic
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
