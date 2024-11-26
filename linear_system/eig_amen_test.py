import copy
import sys
import os

import numpy as np

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import tt_rank_reduce, tt_rl_orthogonalise
from src.tt_eig import tt_max_eig, tt_min_eig


s_matrix_tt = tt_random_gaussian([4, 4], shape=(2, 2))
s_matrix_tt = tt_rl_orthogonalise(tt_rank_reduce(tt_add(s_matrix_tt, tt_transpose(s_matrix_tt)), err_bound=0))

k_matrix_tt = tt_random_gaussian([4, 4], shape=(2, 2))
k_matrix_tt = tt_rl_orthogonalise(tt_rank_reduce(tt_add(s_matrix_tt, tt_transpose(s_matrix_tt)), err_bound=0))

s_matrix_tt = tt_add(s_matrix_tt, k_matrix_tt)


print("Real eig: ", np.max(np.linalg.eigvals(tt_matrix_to_matrix(copy.deepcopy(s_matrix_tt)))))
print("Real eig: ", np.min(np.linalg.eigvals(tt_matrix_to_matrix(copy.deepcopy(s_matrix_tt)))))

eig_val, eig_vec, res = tt_max_eig(copy.deepcopy(s_matrix_tt), verbose=True)
print("Res: ", res)
print("Max Eig val: ", eig_val)

eig_val, eig_vec, res = tt_min_eig(copy.deepcopy(s_matrix_tt), verbose=True)
print("Res: ", res)
print("Min Eig val: ", eig_val)
