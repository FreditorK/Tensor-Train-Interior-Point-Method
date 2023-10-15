import os
import sys

sys.path.append(os.getcwd() + '/../')
from time import time
from src.utils import *
from src.optimiser import ILPSolver

const_space = ConstraintSpace()
head_c1 = const_space.Atom("head(c_1)")
tail_c1 = const_space.Atom("tail(c_1)")
tail_c2 = const_space.Atom("tail(c_2)")
head_c2 = const_space.Atom("head(c_2)")
h = const_space.Hypothesis("h_0")
np.random.seed(7)


def symmetry_objective(tt_train):
    return 1-tt_inner_prod(tt_train, [jnp.swapaxes(t, 0, -1) for t in tt_train[::-1]])


const_space.for_all(h >> (head_c1 ^ tail_c1))
const_space.there_exists(~(h << (head_c1 ^ tail_c1)))
opt = ILPSolver(const_space, objective=symmetry_objective)
t_1 = time()
opt.solve()
t_2 = time()
for h in const_space.hypotheses:
    print(f"Conjunctive Normal Form: {h}{const_space.atoms} =", h.to_CNF(), flush=True)
    print(f"TT-rank: {h}: {tt_rank(h.value)}")
    print(f"Objective value: {symmetry_objective(h.value)}")
print(f"Total time taken: {t_2 - t_1}s.")
