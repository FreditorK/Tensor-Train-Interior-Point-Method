from time import time
from utils import *
from optimiser import ILPSolver, AnswerSetSolver

const_space = ConstraintSpace()
head_c1 = const_space.Atom("head(c_1)")
tail_c1 = const_space.Atom("tail(c_1)")
head_c2 = const_space.Atom("head(c_2)")
tail_c2 = const_space.Atom("tail(c_2)")
h = const_space.Hypothesis("h_0")


def coin_symmetry(tt_train):
    return 1-tt_shared_influence(tt_train, np.array([0, 1, 2, 3]))


const_space.for_all(h >> (head_c1 ^ tail_c1))
opt = ILPSolver(const_space, objective=coin_symmetry)
t_1 = time()
hypothesis = opt.solve()
t_2 = time()
print("TT-rank: ", [f"{h}: {tt_rank(h.value)}" for h in const_space.hypotheses])
print(f"Conjunctive Normal Form: h{const_space.atoms} =", h.to_CNF(), flush=True)
print(f"Total time taken: {t_2 - t_1}s.")
