from time import time
from utils import *
from optimiser import ILPSolver, AnswerSetSolver

const_space = ConstraintSpace()
head_c1 = const_space.Atom("head(c_1)")
tail_c1 = const_space.Atom("tail(c_1)")
head_c2 = const_space.Atom("head(c_2)")
tail_c2 = const_space.Atom("tail(c_2)")
h = const_space.Hypothesis()

a = TTExpression.from_expression((~(h ^ (tail_c1 & head_c1))) | ~(h ^ (tail_c2 << head_c1)))
print(len(a.cores), a.substituted)
b = TTExpression.from_expression((tail_c2 << head_c1))
h.value = b.cores
print(tt_leading_entry(h.substitute_into(a).cores))
#b = TTExpression(tt_permute(a.cores, axes=[(0, 2)]), const_space)
#c = TTExpression([t[:, [1, 0], :] for t in a.cores], const_space)
#print(tt_to_tensor(tt_permute(a.cores)))
#print(tt_to_tensor(tt_bool_op(a.cores)))
#print(b.to_CNF())
#print(c.to_CNF())