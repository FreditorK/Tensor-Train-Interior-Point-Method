import copy
import sys
import os

import numpy as np

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import _tt_core_collapse, _tt_lr_random_orthogonalise, \
    tt_randomised_min_eigentensor, tt_rank_reduce, _tt_generalised_nystroem, _tt_mat_core_collapse
from src.tt_ipm import tt_psd_step
from src.tt_eig import tt_max_eig, tt_min_eig

np.random.seed(2)

s_matrix_tt = tt_random_gaussian([4, 4], shape=(2, 2))
s_matrix_tt = tt_rank_reduce(tt_add(s_matrix_tt, tt_transpose(s_matrix_tt)), err_bound=0)

print("Real eig: ", np.max(np.linalg.eigvals(tt_matrix_to_matrix(copy.deepcopy(s_matrix_tt)))))
print("Real eig: ", np.min(np.linalg.eigvals(tt_matrix_to_matrix(copy.deepcopy(s_matrix_tt)))))

eig_val, eig_vec, res = tt_max_eig(copy.deepcopy(s_matrix_tt), verbose=True)
print("Res: ", res)
print("Max Eig val: ", eig_val)

eig_val, eig_vec, res = tt_min_eig(copy.deepcopy(s_matrix_tt), verbose=True)
print("Res: ", res)
print("Min Eig val: ", eig_val)
