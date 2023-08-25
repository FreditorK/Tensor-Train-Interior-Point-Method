from time import time

import numpy as np

from utils import *
from optimiser import ILPSolver, AnswerSetSolver

const_space = ConstraintSpace()
head_c1 = const_space.Atom("head(c_1)")
tail_c1 = const_space.Atom("tail(c_1)")
head_c2 = const_space.Atom("head(c_2)")
tail_c2 = const_space.Atom("tail(c_2)")
h = const_space.Hypothesis()

a = (tail_c1 | tail_c2) & (head_c1 | ~tail_c2) & (head_c2 | ~head_c1 | ~tail_c1)
b = (head_c1 | tail_c1) & (tail_c1 | ~tail_c2) & (tail_c2 | ~tail_c1)
c = (tail_c1 | ~tail_c2) & (head_c1 | head_c2 | tail_c2) & (~head_c1 | ~tail_c2) & (tail_c2 | ~head_c2 | ~tail_c1)
d = (head_c2 | ~tail_c2) & (head_c1 | ~head_c2 | ~tail_c1) & (tail_c1 | ~head_c1 | ~tail_c2)
e = (tail_c1 | tail_c2) & (tail_c2 | ~head_c1)
arr = [tt_add_noise(a.to_tt_train(), noise_radius=0.1, target_ranks=tt_ranks(a.to_tt_train())), tt_add_noise(b.to_tt_train(), noise_radius=0.1, target_ranks=tt_ranks(b.to_tt_train()))]
#arr = [a.to_tt_train(), b.to_tt_train()]
for tt in arr:
    print("-----")
    print(tt_fast_to_tensor(tt_walsh_op(tt)))
    ta = tt_rank_round(tt_walsh_op(tt))
    print("ii")
    print(tt_fast_to_tensor(ta))
"""
m = a & h
t = TTExpression(m.to_tt_train(), const_space)
print(t.to_CNF())
for k in [b, c, d, e]:
    m = (m | (k & h))
    t = TTExpression(m.to_tt_train(), const_space)
    print(t.to_CNF())
"""
"""
l = (head_c2 | ~tail_c2) & (head_c1 | ~head_c2 | ~tail_c1) & (tail_c1 | ~head_c1 | ~tail_c2)
#for k in [a, b, c, d, e]:
#    print(tt_inner_prod(l.to_tt_train(), k.to_tt_train()))
m = a & l
for k in [b, c, d, e]:
    m = (m | (k & l))
    print(tt_inner_prod(l.to_tt_train(), m.to_tt_train()))
    t = TTExpression(m.to_tt_train(), const_space)
    print(t.to_CNF())
"""
#a = a.to_tt_train()
#print(tt_inner_prod(a, d.to_tt_train()))
#b = b.to_tt_train()
#print(tt_inner_prod(b, d.to_tt_train()))
#c = c.to_tt_train()
#print(tt_inner_prod(c, d.to_tt_train()))
#d = d.to_tt_train()

#a = a + [np.array([1, 0]).reshape(1, 2, 1), np.array([1, 0]).reshape(1, 2, 1)]
#b = b + [np.array([1, 0]).reshape(1, 2, 1), np.array([0, 1]).reshape(1, 2, 1)]
#c = c + [np.array([0, 1]).reshape(1, 2, 1), np.array([0, 1]).reshape(1, 2, 1)]
#f = tt_rank_reduce(tt_add(tt_add(a, b), c))
#print(tt_fast_to_tensor(tt_partial_inner_prod(f, d)))
"""
d = d + [np.array([0, 1]).reshape(1, 2, 1), np.array([1, 0]).reshape(1, 2, 1)]
f = tt_rank_reduce(tt_add(tt_add(a, b), tt_add(c, d)))
print([t.shape for t in f])
f = f[:-2] + [np.einsum("ldr, rk -> ldk", f[-2], f[-1][:, 0, :])]
a = f[:-2] + [np.einsum("ldr, rk -> ldk", f[-2], f[-1][:, 1, :])]
print(TTExpression(a, const_space).to_CNF())
"""


