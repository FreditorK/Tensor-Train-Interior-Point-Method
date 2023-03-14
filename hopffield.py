# imports
from copy import deepcopy

import numpy as np
# for visualization


class Hopfield_Net:  # network class
    # init ialize network variables and memory
    def __init__(self, input):

        # patterns for network training / retrieval
        self.memory = np.array(input)
        # single vs. multiple memories
        if self.memory.size > 1:
            self.n = self.memory.shape[1]
        else:
            self.n = len(self.memory)
        # network construction
        self.weights = np.zeros((self.n, self.n))  # weights vector
        self.energies = []  # container for tracking of energy

    def network_learning(self):  # learn the pattern / patterns
        self.weights = (1 / self.memory.shape[0])*sum((np.outer(self.memory[i], self.memory[i]))**2 for i in range(self.memory.shape[0]))  # hebbian learning
        np.fill_diagonal(self.weights, 0)

    def rect_energy_function(self, x, pow):
        if x >= 0:
            return x**pow
        return 0

    def update_network_state(self, state):  # update network
        for i in range(self.memory.shape[0]):
            state[i] = 0
            state[i] = np.sign(
                sum(
                    self.rect_energy_function(self.memory[mu, i] + self.memory[mu] @ state, 4)
                    - self.rect_energy_function(-self.memory[mu, i] + self.memory[mu] @ state, 4)
                    for mu in range(self.memory.shape[0])
                )
            )
        return state

    def compute_energy(self, state):  # compute energy
        return -0.5 * np.dot(np.dot(state.T, self.weights), state)


net = Hopfield_Net(np.array([[1, 1, 1],
                             [1, 1, -1]]))

net.network_learning()
state = np.array([-1, -1, -1]).reshape(-1, 1)
for _ in range(10):
    print(state.flatten())
    state = net.update_network_state(state)
print(state.flatten())