from time import time
from utils import *
from optimiser import ILPSolver, AnswerSetSolver

const_space = ConstraintSpace()
head_c1 = const_space.Atom("head(c_1)")
tail_c1 = const_space.Atom("tail(c_1)")
head_c2 = const_space.Atom("head(c_2)")
tail_c2 = const_space.Atom("tail(c_2)")
h = const_space.Hypothesis()

a = ((~(h ^ (tail_c1 & head_c1))) | ~(h ^ (tail_c2 << head_c1))).to_tt_train()
c = ((tail_c1 & head_c2) | ~(h | head_c1)).to_tt_train()
a = tt_normalise(tt_hadamard(a, c))
print([t.shape for t in a])
b = tt_randomise_orthogonalise(a, [2, 5, 5, 2])
print([t.shape for t in b])
print(tt_inner_prod(a, b))