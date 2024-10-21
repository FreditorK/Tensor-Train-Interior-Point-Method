import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import _tt_core_collapse, _tt_lr_random_orthogonalise, \
    tt_randomised_min_eigentensor, tt_rank_reduce, _tt_generalised_nystroem, _tt_mat_core_collapse
from src.tt_ineq_check import tt_is_geq

np.random.seed(4)

s_matrix_tt = tt_random_gaussian([4, 4], shape=(2, 2))
s_matrix_tt = tt_rank_reduce(tt_add(s_matrix_tt, tt_transpose(s_matrix_tt)), err_bound=0)

x_tt = tt_random_gaussian([2, 2], shape=(2,))
b_tt = tt_scale(-0.11, [np.ones((1, 2, 1)) for _ in range(3)])

check, val = tt_is_geq(s_matrix_tt, x_tt, b_tt)
print(check, val)
print(np.round(tt_to_tensor(tt_sub(tt_matrix_vec_mul(s_matrix_tt, x_tt), b_tt)), decimals=5))

