import sys
import os

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_ops import *
from src.tt_ops import _tt_core_collapse

np.random.seed(4)

linear_op = tt_random_gaussian([4, 5, 8, 6, 4], shape=(2, 2))
linear_op = tt_rank_retraction(tt_add(linear_op, tt_transpose(linear_op)), [2, 3, 3, 3, 2])
linear_op = tt_normalise(linear_op)

eig_vec_tt = tt_normalise([np.random.randn(1, 2, 1) for _ in range(len(linear_op))])

tt_vec = tt_matrix_vec_mul(linear_op, eig_vec_tt)

tt_eig, tt_eig_val = tt_max_eigentensor(linear_op, num_iter=1500, tol=1e-8)
print(tt_eig_val)


matrices = [_tt_core_collapse(c, c) for c in tt_vec]
M = np.outer(matrices[-1], matrices[0]) @ np.linalg.multi_dot(matrices[1:-1])
M = M / np.linalg.norm(M)
print(np.linalg.eigvals(M))

