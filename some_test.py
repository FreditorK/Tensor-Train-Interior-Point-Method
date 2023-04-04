import numpy as np

from operators import D_func
from utils import *
from tt_op import tt_extract_seq
from optimiser import AnswerSetSolver

vocab_size = 4
x = Atom(vocab_size, "x")
y = Atom(vocab_size, "y")
z = Atom(vocab_size, "z")
w = Atom(vocab_size, "w")
e_tt_1 = (x & (y | z | w)).to_tt_train()
e_tt_2 = ((x & (y | z)) >> w).to_tt_train()
e_tt_3 = (x & (y | z) & w).to_tt_train()
e_tt_4 = (x ^ w).to_tt_train()
#asp_solver = AnswerSetSolver([x, y, z, w])
#print(asp_solver.get_minimal_answer_set(e_tt_1))
#print(asp_solver.get_minimal_answer_set(e_tt_2, x=1, y=1))
#print(asp_solver.get_minimal_answer_set(e_tt_3))
#print(asp_solver.get_minimal_answer_set(e_tt_4))
#print(jnp.sum(tt_to_tensor([np.array([1.0, 0.0]).reshape(1, 2, 1)]+[np.array([0.1, 0.9]).reshape(1, 2, 1) for _ in range(5)])))
ps = np.array([0.5, 0.1, 1.0, 1.0])
e_tt_1_meas = tt_measure(e_tt_1, ps)  # P(x=1| x=1), P(x=-1| x=-1)
#print(tt_to_tensor(e_tt_1))
#print(tt_to_tensor(e_tt_1_meas))
weighted_tensor = tt_bool_op_inv(e_tt_1_meas)
#weighted_tensor[0] *= 1/jnp.sqrt(tt_inner_prod(weighted_tensor, weighted_tensor))
#print(tt_to_tensor(weighted_tensor))
print(tt_inner_prod(weighted_tensor, weighted_tensor))
#unweighted_tensor = tt_measure_inv(tt_bool_op(weighted_tensor), ps, qs)
print(tt_to_tensor(e_tt_1))
print(tt_to_tensor(weighted_tensor))
print(tt_inner_prod(weighted_tensor, [np.array([1, 1]).reshape(1, 2, 1), np.array([1, -1]).reshape(1, 2, 1), np.array([1, -1]).reshape(1, 2, 1), np.array([1, -1]).reshape(1, 2, 1)]))
print(tt_inner_prod(e_tt_1, [np.array([1, 0.5]).reshape(1, 2, 1), np.array([1, -0.1]).reshape(1, 2, 1), np.array([1, -1]).reshape(1, 2, 1), np.array([1, -1]).reshape(1, 2, 1)]))