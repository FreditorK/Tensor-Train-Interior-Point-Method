from functools import reduce
from time import time

import numpy as np

from utils import *
from optimiser import ILPSolver, AnswerSetSolver

vocab_size = 3
atoms = np.array(generate_atoms(vocab_size))
"""
set_1 = [0, 2, 3, 6]
set_2 = [1, 6]
set_3 = [4]
set_U = [4, 6]
s_1 = reduce(lambda x, y: x | y, atoms[set_1])
s_2 = reduce(lambda x, y: x | y, atoms[set_2])
s_3 = atoms[set_3]
s_u = reduce(lambda x, y: x | y, atoms[set_U])
"""
h_1 = Hypothesis()
#h_2 = Hypothesis()
#h_3 = Hypothesis()
a = (atoms[0] & ~h_1).to_tt_train()
h = (atoms[1] | atoms[2]).to_tt_train()
h[-2] = np.einsum("ldr, rk -> ldk", h[-2], h[-1][:, 0, :])
h.pop()
h_1.value = h
k = h_1.substitute_into(a)
k = [np.power(c, 1) for c in k]
print(len(k))
print("Subsituted in:", get_CNF(atoms, k))
print(np.round(tt_to_tensor(k), decimals=2))
#print(tt_to_tensor(k))
solution = (atoms[0] & ~(atoms[1] | atoms[2])).to_tt_train()
solution[-2] = np.einsum("ldr, rk -> ldk", solution[-2], solution[-1][:, 0, :])
solution.pop()
print("Truth", get_CNF(atoms, solution))
print(tt_inner_prod(solution, k))
#const_space = ConstraintSpace(vocab_size)