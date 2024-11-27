import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import tt_rank_reduce, tt_matrix_vec_mul
from src.tt_amen import tt_amen

np.random.seed(4)
ranks = [2, 9, 15, 26, 7]
s_matrix_tt = tt_rank_reduce(tt_random_gaussian(ranks, shape=(2, 2)))
#s_matrix_tt = tt_rank_reduce(tt_add(s_matrix_tt, tt_transpose(s_matrix_tt)), err_bound=0)

b_tt = tt_rank_reduce(tt_matrix_vec_mul(s_matrix_tt, tt_random_gaussian(ranks, shape=(2,))))
sol, res = tt_amen(s_matrix_tt, b_tt, kickrank=4, verbose=True)
#res_tt = tt_sub(tt_matrix_vec_mul(s_matrix_tt, sol), b_tt)
#print(tt_inner_prod(res_tt, res_tt))


