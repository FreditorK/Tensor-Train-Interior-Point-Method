import sys
import os
import time
from copy import copy

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_op import *
from sklearn.utils.extmath import randomized_range_finder
from src.tt_op import _tt_op_op_collapse

@dataclass
class Config:
    num_columns = 2
    tt_length = 6
    tt_max_rank = 5

np.random.seed(10)


L = np.random.randn(3, 2, 2, 4)
U, S, V_T = np.linalg.svd(L.reshape(-1, L.shape[-1]))
print(S)

#L_sq_0 = sum(sum(np.kron(L[:, 0, i_2], L[:, i_2, i_1]) for i_1 in [0, 1]) for i_2 in [0, 1])
#L_sq_1 = sum(sum(np.kron(L[:, 1, i_2], L[:, i_2, i_1]) for i_1 in [0, 1]) for i_2 in [0, 1])
#print(L_sq_0.shape, L_sq_1.shape)
#L_sq_concat = np.concatenate((L_sq_0, L_sq_1), axis=1)
L_sq_concat = _tt_op_op_collapse(L, L)
print(L_sq_concat.shape)
U, S, V_T = np.linalg.svd(L_sq_concat.reshape(-1, L_sq_concat.shape[-1]))
print(S)
Q, R = np.linalg.qr(L_sq_concat.reshape(-1, L_sq_concat.shape[-1]))
print(Q.shape, R.shape)
L_sq_concat = L_sq_concat.reshape(-1, L_sq_concat.shape[-1])
#rngeQ = randomized_range_finder(L_sq_concat, size=15, n_iter=10, random_state=9)
#print(np.sum(rngeQ @ rngeQ.T @ L_sq_concat -L_sq_concat))

tensor = tt_random_matrix([4, 5, 4, 3])
z = [6*np.random.rand(1, 2, 19)] + [5*np.random.randn(19, 2, 19) + 0.3*np.random.randn(19, 2, 19) + np.random.randint(5) for _ in tensor[1:-2]] + [np.random.randn(19, 2, 23) + 6*(np.random.randn(19, 2, 23)-0.2)] + [np.random.rand(23, 2, 1)]
y = [np.round(10*np.random.randn(1, 2, 11))] + [np.round(5*np.random.rand(11, 2, 11)) for _ in tensor[1:-2]] + [np.random.randn(11, 2, 2)] + [np.random.randn(2, 2, 1)]
x = tt_add(tt_hadamard(z,y), z)
print("A", [c.shape for c in tt_rank_reduce(x, tt_bound=0)])
comp = tt_linear_op(tensor, x)
x = tt_rank_reduce(comp, tt_bound=0)
print("B", [c.shape for c in x])
comp = tt_linear_op(tensor, x)
x = tt_rank_reduce(comp, tt_bound=0)
print("B_1", [c.shape for c in x])
comp = tt_linear_op(tensor, x)
x = tt_rank_reduce(comp, tt_bound=0)
print("B_2", [c.shape for c in x])
comp = tt_linear_op(tensor, x)
x = tt_rank_reduce(comp, tt_bound=0)
print("B_3", [c.shape for c in x])

# TODO: This below is equal to taking the inner product between two tensor matrices
"""
print(tt_inner_prod(z_sym, y_sym))
k = tt_linear_op(z_sym, y)
print(tt_inner_prod(y, k))
"""
