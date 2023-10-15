import os
import sys

sys.path.append(os.getcwd() + '/../')
from time import time
from src.utils import *
from src.optimiser import ILPSolver

const_space = ConstraintSpace()
x = const_space.Atom("x")
y = const_space.Atom("y")
z = const_space.Atom("z")
w = const_space.Atom("w")
u = const_space.Atom("u")
h = const_space.Hypothesis("h_0")
const_space.for_all(h << (x & y & z))
const_space.for_all(h >> x)
const_space.there_exists(h << (x & w))
const_space.there_exists(h << (x << (y & z)))
const_space.there_exists(h << (x & ~z & ~y))
const_space.there_exists(h | ~(u & w))
opt = ILPSolver(const_space)
t_1 = time()
opt.solve()
t_2 = time()
for h in const_space.hypotheses:
    print(f"TT-rank: {h}: {tt_rank(h.value)}")
    print(f"Conjunctive Normal Form: {h}{const_space.atoms} =", h.to_CNF(), flush=True)
print(f"Total time taken: {t_2 - t_1}s.")
