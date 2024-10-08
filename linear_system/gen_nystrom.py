import copy
import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import _tt_core_collapse, _tt_lr_random_orthogonalise, \
    tt_randomised_min_eigentensor, tt_rank_reduce, _tt_generalised_nystroem, _tt_mat_core_collapse
from src.tt_ipm import tt_psd_step

np.random.seed(69)

psd_matrix_tt = tt_random_gaussian([4, 4], shape=(2, 2))
psd_matrix_tt = tt_rank_reduce(tt_mat_mat_mul(psd_matrix_tt, tt_transpose(psd_matrix_tt)), err_bound=0)
Delta_tt = tt_random_gaussian([4, 4], shape=(2, 2))
Delta_tt = tt_rank_reduce(tt_add(Delta_tt, tt_transpose(Delta_tt)), err_bound=0)

alpha = 0.5
psd_matrix_tt_add = tt_add(psd_matrix_tt, tt_scale(-alpha, Delta_tt))


_, eig_val_2 = tt_psd_step(copy.deepcopy(psd_matrix_tt), copy.deepcopy(Delta_tt), block_size=5, num_iter=5000)
print("Eig val part: ", eig_val_2)
print("Real eig: ", np.min(np.linalg.eigvals(tt_matrix_to_matrix(tt_add(copy.deepcopy(psd_matrix_tt), copy.deepcopy(Delta_tt))))))
print("Real eig: ", np.min(np.linalg.eigvals(tt_matrix_to_matrix(tt_add(copy.deepcopy(psd_matrix_tt), tt_scale(0.5**(9), copy.deepcopy(Delta_tt)))))))
