import copy
import sys
import os

import numpy as np

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import tt_rank_reduce, tt_rl_orthogonalise, tt_mat_mat_mul, tt_rank_retraction
from src.tt_eig import tt_max_eig, tt_min_eig, tt_null_space


n = 32
r = 31
A = np.random.randn(n, r) @ np.random.randn(r, n)
s_matrix_tt = tt_matrix_svd(A, err_bound=1e-10)

global_res, null_vec, res = tt_null_space(copy.deepcopy(s_matrix_tt), verbose=True)

print("Rank  of A: ")
print(tt_ranks(s_matrix_tt))
print(np.sum(np.linalg.svdvals(tt_matrix_to_matrix(s_matrix_tt)) > 1e-10))

print("Local Res: ", res)
print("Global Res: ", global_res)
print(tt_ranks(tt_rank_reduce(tt_mat(null_vec), eps=1e-10)))
null_mat = tt_mat(null_vec)
null_res_mat = tt_mat_mat_mul(s_matrix_tt, null_mat)
#print(np.round(tt_matrix_to_matrix(null_res_mat), decimals=4))
#print(np.round(tt_matrix_to_matrix(null_mat), decimals=2))
print(np.sum(np.linalg.svdvals(tt_matrix_to_matrix(null_mat)) > 1e-10))
