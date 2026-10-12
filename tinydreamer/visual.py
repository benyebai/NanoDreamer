"""Small NumPy autoencoder for MinAtar object-channel observations."""

import numpy as np
from mininn.activations import Tanh
from mininn.linear import Linear
from mininn.sequential import Sequential
from tensorgrad import Tensor


class VisualAutoencoder:
    def __init__(self, observation_shape=(10, 10, 6), hidden_size=96, latent_size=32):
        self.observation_shape = tuple(observation_shape)
        # in this case 10x10x6
        self.input_size = int(np.prod(self.observation_shape))
        self.latent_size = latent_size
        self.encoder = Sequential(
            Linear(self.input_size, hidden_size),
            Tanh(),
            Linear(hidden_size, latent_size),
        )
        self.decoder = Sequential(
            Linear(latent_size, hidden_size),
            Tanh(),
            Linear(hidden_size, self.input_size),
        )

    def encode(self, observations):
        # first reshape from (batches, 10, 10, 6) -> (batches, 600) -> into encoder
        return self.encoder(observations.reshape((-1, self.input_size)))

    def decode(self, latents):
        # decoder(latents) -> (batches, 600) that came out of the decoder and shape back to (batches, 10, 10, 6)
        return self.decoder(latents).reshape((-1, *self.observation_shape))

    def __call__(self, observations):
        return self.decode(self.encode(observations))

    def parameters(self):
        return self.encoder.parameters() + self.decoder.parameters()

    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()


# to influence which ones we care a lot about, so give it a fat error multiplyer
POSITIVE_WEIGHTS = np.array([20.0, 2.0, 2.0, 2.0, 80.0, 80.0]).reshape(1, 1, 1, 6)


def weighted_binary_cross_entropy(logits, targets, positive_weight=POSITIVE_WEIGHTS):
    probabilities = logits.sigmoid()
    eps = 1e-7
    return -(
        # if the real value was 1, then the probability thats its 1 is just (P)
        # if the real value was 0, then the probability thats its 0 is just (1 - P)
        Tensor(positive_weight) * targets * (probabilities + eps).log()
        + (1 - targets) * (1 - probabilities + eps).log()
    ).mean()


# test how well our reconstruction did
def reconstruction_metrics(logits, targets, threshold=0.5):
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
    predicted = probabilities >= threshold
    truth = targets.astype(bool)
    tp = np.logical_and(predicted, truth).sum()
    fp = np.logical_and(predicted, ~truth).sum()
    fn = np.logical_and(~predicted, truth).sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pixel_accuracy": float((predicted == truth).mean()),
        "foreground_precision": float(precision),
        "foreground_recall": float(recall),
        "foreground_f1": float(f1),
    }
