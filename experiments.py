from time import time
from utils import *
from optimiser import ILPSolver, AnswerSetSolver

const_space = ConstraintSpace()
head_c1 = const_space.Atom("head(c_1)")
tail_c1 = const_space.Atom("tail(c_1)")
head_c2 = const_space.Atom("head(c_2)")
tail_c2 = const_space.Atom("tail(c_2)")

a = TTExpression.from_expression((head_c1 ^ tail_c1) & (head_c2 ^ tail_c2))
b = TTExpression(tt_permute(a.cores, axes=[(0, 2)]), const_space)
#c = TTExpression([t[:, [1, 0], :] for t in a.cores], const_space)
#print(tt_to_tensor(tt_permute(a.cores)))
print(a.to_CNF())
print(b.to_CNF())
#print(c.to_CNF())