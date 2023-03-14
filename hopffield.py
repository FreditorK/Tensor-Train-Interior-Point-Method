# imports
from copy import deepcopy

import numpy as np
# for visualization


class Hopfield_Net:  # network class
    # init ialize network variables and memory
    def __init__(self, input):

        # patterns for network training / retrieval
        self.memory = input
        # single vs. multiple memories
        if self.memory.size > 1:
            self.n = self.memory.shape[1]
        else:
            self.n = len(self.memory)
        # network construction
        self.energies = []  # container for tracking of energy

    def rect_energy_function(self, x, pow):
        if x >= 0:
            return x**pow
        return 0

    def update_network_state(self, state):  # update network
        for i in range(self.n):
            state[i] = 0
            state[i] = np.sign(
                sum(
                    self.rect_energy_function(self.memory[mu, i] + self.memory[mu] @ state, 4)
                    - self.rect_energy_function(-self.memory[mu, i] + self.memory[mu] @ state, 4)
                    for mu in range(self.memory.shape[0])
                )
            )
        return state


net = Hopfield_Net(np.array([[1, 1, 1, -1, -1, -1], # need to append inverted sequence
                             [1, 1, -1, -1, -1, 1],
                             [1, -1, 1, -1, 1, -1]]))

state = np.array([-1, -1, 1, 1, 1, -1])
for _ in range(5):
    state = net.update_network_state(state)
print(state.flatten())