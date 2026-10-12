import numpy as np
from mininn.activations import Tanh
from mininn.linear import Linear
from mininn.sequential import Sequential
from tensorgrad import Tensor


class Critic:
    def __init__(self, seed=0, input_size=27):
        np.random.seed(seed)
        self.input_size = input_size
        self.network = Sequential(
            Linear(input_size, 32),
            Tanh(),
            Linear(32, 1),
        )

    def forward(self, observation):
        x = Tensor(observation.reshape(1, self.input_size))
        return self.network(x)

    def predict(self, observation):
        return float(self.forward(observation).data[0, 0])

    def parameters(self):
        return self.network.parameters()
