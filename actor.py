"""The policy network for choosing one of the four maze actions."""

import numpy as np
from mininn.activations import Tanh
from mininn.linear import Linear
from mininn.sequential import Sequential
from tensorgrad import Tensor


class Actor:
    def __init__(self, seed=0, input_size=27):
        np.random.seed(seed)
        self.input_size = input_size
        self.network = Sequential(
            Linear(input_size, 32),
            Tanh(),
            Linear(32, 4),
        )
        self.random_generator = np.random.default_rng(seed)

    def forward(self, observation):
        x = Tensor(observation.reshape(1, self.input_size))
        return self.network(x)

    def probabilities(self, observation):
        scores = self.forward(observation)
        probabilities = scores.softmax(axis=1)
        return probabilities.data[0]

    def choose_action(self, observation):
        probabilities = self.probabilities(observation)
        return int(self.random_generator.choice(4, p=probabilities))

    def parameters(self):
        return self.network.parameters()
