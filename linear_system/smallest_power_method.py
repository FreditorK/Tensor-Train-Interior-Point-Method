import sys
import os
import time

import numpy as np

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *

np.random.seed(11)

linear_op = tt_random_gaussian_linear_op([4, 5, 22, 5, 8, 6, 8])
linear_op = tt_rank_reduce(tt_add(linear_op, tt_transpose(linear_op)), tt_bound=1e-6)
t0 = time.time()
tt_eig, tt_eig_val = tt_randomised_min_eigentensor(linear_op, num_iter=1200, tol=1e-6)
t1 = time.time()
print("Linear Op ranks: ", tt_ranks(linear_op))
print("Ranks of eigen tensor: ", tt_ranks(tt_eig))
print("Eigen value: ", tt_eig_val)
print(f"Power Method converged in {t1-t0:.4f}s")
matrix = tt_op_to_matrix(linear_op)
min_eig_val = np.min(np.linalg.eigvals(matrix))
print(np.round(min_eig_val, decimals=4))