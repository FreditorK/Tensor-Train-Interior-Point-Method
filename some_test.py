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
ps = np.array([1.0, 1.0, 1.0, 1.0])
#print(tt_to_tensor(e_tt_1))
#print(tt_to_tensor(e_tt_1_meas))
weighted_tensor = tt_noise_op(e_tt_1, ps)
#weighted_tensor[0] *= 1/jnp.sqrt(tt_inner_prod(weighted_tensor, weighted_tensor))
#print(tt_to_tensor(weighted_tensor))
#unweighted_tensor = tt_measure_inv(tt_bool_op(weighted_tensor), ps, qs)
print(tt_inner_prod(weighted_tensor, [np.array([1, 1]).reshape(1, 2, 1), np.array([1, -1]).reshape(1, 2, 1), np.array([1, -1]).reshape(1, 2, 1), np.array([1, -1]).reshape(1, 2, 1)]))
print(tt_inner_prod(e_tt_1, [np.array([1, 1]).reshape(1, 2, 1), np.array([1, -0.25]).reshape(1, 2, 1), np.array([1, -1]).reshape(1, 2, 1), np.array([1, -1]).reshape(1, 2, 1)]))
print(jnp.sqrt(tt_inner_prod(weighted_tensor, weighted_tensor)))
"""
tens = 2*np.random.rand(2, 2, 2)-1
tens[np.random.randint(0, 2), np.random.randint(0, 2), np.random.randint(0, 2)] = 1.0
tens *= 1/np.sqrt(np.sum(tens*tens))
true = tt_svd(tens)
print(tt_inner_prod(true, true))
print(tt_to_tensor(true))
print(tt_to_tensor(tt_bool_op(true)))
clipped_true = [np.clip(c, a_max=1, a_min=-1) for c in tt_bool_op(true)]
print(tt_to_tensor(clipped_true))

jax.random.PRNGKey(112)
tensor_1 = np.random.rand(2, 2, 2, 2, 2, 2, 2)
tt_1 = tt_svd(tensor_1)
gra_func = D_func(lambda t: tt_rank_loss(t))
tt_2 = [np.ones((1, 2, 1)) for _ in range(7)]
print(tt_rank_loss(tt_1), [c.shape for c in tt_1])
print(tt_rank_loss(tt_2), [c.shape for c in tt_2])
lr = 0.1
for _ in range(50):
    gradient = gra_func(tt_1)
    for i, g in enumerate(gradient):
        tt_1[i] -= lr * g[i]
    lr *= 0.999
tt_1 = tt_rank_reduce(tt_1, tt_bound=1e-4)
print(tt_rank_loss(tt_1), [c.shape for c in tt_1])
"""