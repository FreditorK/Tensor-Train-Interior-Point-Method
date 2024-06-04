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

linear_op = tt_random_gaussian_linear_op([4, 5, 22, 5, 8, 6, 8])
linear_op = tt_scale(9, tt_rank_reduce(tt_add(linear_op, tt_transpose(linear_op)), tt_bound=1e-12))
t0 = time.time()
tt_eig, tt_eig_val = tt_max_eigentensor(linear_op, num_iter=1000, tol=1e-7)
t1 = time.time()
proj_rank = [int(np.ceil(np.sqrt(r))) for r in tt_ranks(linear_op)]
#proj_rank = [r for r in tt_ranks(tt_eig)]
print("Ranks of op tensor: ", tt_ranks(linear_op))
print("Proj rank: ", proj_rank)
sym = tt_scale(tt_eig_val, tt_rank_reduce(tt_outer_product(tt_eig, tt_eig), tt_bound=1e-12))
print("sym", tt_ranks(sym))
sym_2 = tt_rank_reduce(tt_add(linear_op, tt_scale(-1, sym)), tt_bound=1e-12)
print("sym_2", tt_ranks(sym_2))
sym = tt_scale(tt_eig_val, tt_rank_reduce(tt_outer_product(tt_eig, tt_eig), tt_bound=1e-12))
sym_3 = tt_rank_reduce(tt_add(sym_2, sym), tt_bound=1e-12)
print("sym_3", tt_ranks(sym_3))
diff = tt_add(tt_normalise(tt_eig), tt_scale(-1, tt_normalise(tt_eig)))
print("Error", tt_inner_prod(diff, diff))
print("Ranks of eigen tensor: ", tt_ranks(tt_eig))
print("Eigen value: ", tt_eig_val)
print(f"Power Method converged in {t1-t0:.4f}s")
matrix = tt_op_to_matrix(linear_op)
max_eig_val = np.max(np.linalg.eigvals(matrix))
print(np.round(max_eig_val, decimals=4))
