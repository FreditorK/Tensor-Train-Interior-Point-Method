from time import time
from utils import *
from optimiser import ILPSolver, AnswerSetSolver

const_space = ConstraintSpace()
x = const_space.Atom("x")
y = const_space.Atom("y")
z = const_space.Atom("z")
w = const_space.Atom("w")
u = const_space.Atom("u")
h = const_space.Hypothesis("h_0")
const_space.there_exists(h << (x & w))
const_space.for_all(h << (x & y & z))
const_space.there_exists(h << (x << (y & z)))
const_space.for_all(h >> x)
const_space.there_exists(h << (x & ~z & ~y))
const_space.there_exists(h | ~(u & w))
opt = ILPSolver(const_space, objective=tt_nuc_schatten_norm)
t_1 = time()
opt.solve()
t_2 = time()
print("TT-rank: ", [f"{h}: {tt_rank(h.value)}" for h in const_space.hypotheses])
print(h.to_CNF(), flush=True)
print(f"Total time taken: {t_2 - t_1}s.")
