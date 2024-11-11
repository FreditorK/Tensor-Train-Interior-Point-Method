import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import _tt_core_collapse, _tt_lr_random_orthogonalise, \
    tt_randomised_min_eigentensor, tt_rank_reduce, _tt_generalised_nystroem, _tt_mat_core_collapse
from src.tt_amen import tt_amen

np.random.seed(4)

s_matrix_tt = tt_random_gaussian([4, 4], shape=(2, 2))
s_matrix_tt = tt_rank_reduce(tt_add(s_matrix_tt, tt_transpose(s_matrix_tt)), err_bound=0)

b_tt = tt_random_gaussian([4, 4], shape=(2,))
print([c.shape for c in s_matrix_tt])
print([c.shape for c in b_tt])
sol = tt_amen(s_matrix_tt, b_tt, kickrank=2, verbose=True)
res_tt = tt_sub(tt_matrix_vec_mul(s_matrix_tt, sol), b_tt)
print(tt_inner_prod(res_tt, res_tt))


