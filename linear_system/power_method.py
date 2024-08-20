import sys
import os
import time
import numpy as np
import copy

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_ops import *

@dataclass
class Config:
    num_columns = 8
    tt_length = 8
    tt_max_rank = 9

np.random.seed(929)

linear_op = tt_random_gaussian([4, 5, 22, 5, 8, 6, 8], shape=(2, 2))
linear_op = tt_scale(9, tt_rank_reduce(tt_add(linear_op, tt_transpose(linear_op)), err_bound=1e-10))
t0 = time.time()
tt_eig, tt_eig_val = tt_max_eigentensor(linear_op, num_iter=1000, tol=1e-7)
t1 = time.time()
print("Ranks of op tensor: ", tt_ranks(linear_op))
print("Ranks of eigen tensor: ", tt_ranks(tt_eig))
print("Eigen value: ", tt_eig_val)
print(f"Power Method converged in {t1-t0:.4f}s")
matrix = tt_matrix_to_matrix(linear_op)
max_eig_val = np.max(np.linalg.eigvals(matrix))
print(np.round(max_eig_val, decimals=4))
