from time import time
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
"""
m = a & h
t = TTExpression(m.to_tt_train(), const_space)
print(t.to_CNF())
for k in [b, c, d, e]:
    m = (m | (k & h))
    t = TTExpression(m.to_tt_train(), const_space)
    print(t.to_CNF())
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


