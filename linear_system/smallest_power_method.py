import sys
import os
import time

import numpy as np

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *

np.random.seed(190)

linear_op = tt_random_gaussian_linear_op([4, 5, 15, 5, 8, 9, 10, 6, 4])
linear_op = tt_rank_reduce(tt_add(linear_op, tt_transpose(linear_op)), err_bound=1e-10)
t0 = time.time()
tt_eig, tt_eig_val = tt_randomised_min_eigentensor(linear_op, num_iter=1500, tol=1e-8)
t1 = time.time()
print(tt_ranks(tt_rank_reduce(tt_outer_product(tt_eig, tt_eig))))
print("Linear Op ranks: ", tt_ranks(linear_op))
print("Ranks of eigen tensor: ", tt_ranks(tt_eig))
print("Eigen value: ", tt_eig_val)
print(f"Power Method converged in {t1-t0:.4f}s")
matrix = tt_matrix_to_matrix(linear_op)
min_eig_val = np.min(np.linalg.eigvals(matrix))
print(np.round(min_eig_val, decimals=5))