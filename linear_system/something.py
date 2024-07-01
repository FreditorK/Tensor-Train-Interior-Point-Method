import sys
import os
import time
from copy import copy

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_ops import *
from sklearn.utils.extmath import randomized_range_finder
from src.tt_ops import *


@dataclass
class Config:
    num_columns = 2
    tt_length = 6
    tt_max_rank = 5


np.random.seed(42)


def _tt_op_op_collapse(linear_core_op_1, linear_core_op_2):
    return sum([
        np.kron(linear_core_op_1[:, None, i], linear_core_op_2[:, :, None, i])
        for i in range(linear_core_op_2.shape[2])
    ])


V = tt_random_gaussian([2, 3], (2, 2))
V_matrix = tt_op_to_matrix(V)
A = tt_random_gaussian([2, 3], (2, 2))
A_matrix = tt_op_to_matrix(A)

AV_sol_matrix = A_matrix @ V_matrix

AV = tt_linear_op_compose(A, V)
print([c.shape for c in AV])
AV_matrix = tt_op_to_matrix(AV)

print(np.round(AV_sol_matrix, decimals=2))

print(np.round(AV_matrix, decimals=2))

"""

V = tt_random_gaussian([2, 3, 9, 8, 5], (2, 2))
V = tt_add(V, tt_transpose(V))
V = tt_rank_reduce(V)
print(tt_ranks(V))
M = tt_op_to_matrix(V)
eigenvalues, eigenvectors = np.linalg.eig(M)
print(len(eigenvectors))
ran = [np.zeros((1, 2, 2, 1)) for _ in range(len(V))]
for i, vec in enumerate(eigenvectors[:-2]):
    tt_eigenvector = tt_svd(vec.reshape(2, 2, 2, 2, 2, 2))
    ran = tt_add_column(ran, tt_eigenvector, i)
ran = tt_rank_reduce(ran)
ran = tt_linear_op_compose(ran, tt_transpose(ran))
ran = tt_rank_reduce(ran)
print(tt_ranks(ran))
print(np.round(tt_op_to_matrix(ran)))
print(eigenvectors[-1])
"""

"""
A_copy = copy(A)
B = tt_rank_reduce(A, tt_bound=1e-6)
A = tt_generalised_nystroem(A_copy, tt_ranks(A))
diff = tt_sub(B, A)
print(tt_inner_prod(diff, diff))
"""

"""
print(tt_inner_prod(z_sym, y_sym))
k = tt_linear_op(z_sym, y)
print(tt_inner_prod(y, k))
"""
