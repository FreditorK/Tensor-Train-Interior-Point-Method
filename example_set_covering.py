from functools import reduce
from time import time

import numpy as np

from utils import *
from optimiser import ILPSolver, AnswerSetSolver

vocab_size = 7
const_space = ConstraintSpace()
atoms = np.array(const_space.generate_atoms(vocab_size))
h_0 = const_space.Hypothesis()
h_1 = const_space.Hypothesis()
h_2 = const_space.Hypothesis()
set_0 = [0, 2, 3, 6]
set_1 = [1, 6]
set_2 = [4]
set_U = [4, 6]
s_0 = reduce(lambda x, y: x | y, atoms[set_0])
s_1 = reduce(lambda x, y: x | y, atoms[set_1])
s_2 = atoms[set_2].item()
s_u = reduce(lambda x, y: x | y, atoms[set_U])
const_space.for_all(s_u << (h_0 | h_1 | h_2))
const_space.for_all(~((h_0 | s_0) ^ s_0))
const_space.for_all(~((h_1 | s_1) ^ s_1))
const_space.for_all(~((h_2 | s_2) ^ s_2))

tt_false = tt_mul_scal(-1, tt_leading_one(const_space.atom_count))
set_0_caridinalty = len(set_0) / const_space.atom_count
set_1_caridinalty = len(set_1) / const_space.atom_count
set_2_caridinalty = len(set_2) / const_space.atom_count


def set_weights(tt_0, tt_1, tt_2):
    return -(
        set_0_caridinalty*tt_inner_prod(tt_false, tt_0)
        + set_1_caridinalty*tt_inner_prod(tt_false, tt_1)
        + set_2_caridinalty*tt_inner_prod(tt_false, tt_2)
    )


opt = ILPSolver(const_space, objective=set_weights)
t_1 = time()
opt.solve()
t_2 = time()
print("TT-rank: ", [f"{h}: {tt_rank(h.value)}" for h in const_space.hypotheses])
print(f"Conjunctive Normal Form: h_0{const_space.atoms[set_0]} =", h_0.to_CNF(), flush=True)
print(f"Conjunctive Normal Form: h_1{const_space.atoms[set_1]} =", h_1.to_CNF(), flush=True)
print(f"Conjunctive Normal Form: h_2{const_space.atoms[set_2]} =", h_2.to_CNF(), flush=True)
print(f"Total time taken: {t_2 - t_1}s.")
