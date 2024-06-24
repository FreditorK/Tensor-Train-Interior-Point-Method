import sys
import os
import time
from copy import copy

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_ops import *
from sklearn.utils.extmath import randomized_range_finder
from src.tt_ops import _tt_op_op_collapse


@dataclass
class Config:
    num_columns = 2
    tt_length = 6
    tt_max_rank = 5


np.random.seed(333)

"""
X = tt_rl_orthogonalise(tt_random_gaussian_linear_op([2]))
X = tt_add(X, tt_transpose(X))
one = tt_one(4)
one = [core_bond(c_1, c_2) for c_1, c_2 in zip(one[:-1], one[1:])]
G = tt_scale(1 / 2, tt_add(tt_random_graph([4]), one))
X_mask = tt_hadamard(G, X)
print(np.round(tt_op_to_matrix(G), decimals=1))
print(np.round(tt_op_to_matrix(X_mask), decimals=3))

G_op = tt_mask_to_linear_op(G)
X_mask = tt_eval_constraints(G_op, X)
X_mask = tt_rank_reduce(X_mask)
print([c.shape for c in X_mask])
X_mask = [X_mask[0].reshape(1, 2, 2, 2), X_mask[1].reshape(2, 2, 2, 1)]

print(np.round(tt_op_to_matrix(X_mask), decimals=3))
print([c.shape for c in X_mask])

# TODO: This below is equal to taking the inner product between two tensor matrices

"""

A = tt_random_gaussian_linear_op([2, 2])
M_1 = tt_op_to_matrix(A)
A = tt_linear_op_compose(A, tt_transpose(A))
A = tt_rank_reduce(A, tt_bound=1e-4)
M = tt_op_to_matrix(A)
#M = M_1 @ M_1.T
#print(tt_ranks(A))
print(np.linalg.eigvals(A[1][:, 0, 0, :] @ A[1][:, 0, 0, :].T))
print(np.linalg.eigvals(M))

"""
print(tt_inner_prod(z_sym, y_sym))
k = tt_linear_op(z_sym, y)
print(tt_inner_prod(y, k))
"""
