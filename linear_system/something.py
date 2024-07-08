import sys
import os
import time
from copy import copy

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_ops import *
from src.tt_ops import *


@dataclass
class Config:
    num_columns = 2
    tt_length = 6
    tt_max_rank = 5


np.random.seed(42)

V = tt_random_binary([4, 3, 5])
print(np.round(tt_to_tensor(V), decimals=3))
idx = np.array(tt_argmax(V), dtype=int)
print(idx)
vs = [v[:, i] for i, v in zip(idx, V)]
result = np.linalg.multi_dot(vs)
print(result)



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
