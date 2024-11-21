import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.ops import *
from src.tt_ops import _tt_core_collapse, _tt_lr_random_orthogonalise, \
    tt_randomised_min_eigentensor, tt_rank_reduce, _tt_generalised_nystroem, _tt_mat_core_collapse, tt_matrix_vec_mul
from src.tt_amen import tt_amen, tt_block_amen

np.random.seed(4)
ranks = [2]
block_1_tt = tt_rank_reduce(tt_random_gaussian(ranks, shape=(2, 2)))
block_2_tt = tt_rank_reduce(tt_random_gaussian(ranks, shape=(2, 2)))
block_matrix_tt = {(0, 0): block_1_tt, (1, 1): block_2_tt}

b_1_tt = tt_rank_reduce(tt_matrix_vec_mul(block_1_tt, tt_random_gaussian(ranks, shape=(2,))))
b_2_tt = tt_rank_reduce(tt_matrix_vec_mul(block_2_tt, tt_random_gaussian(ranks, shape=(2,))))
block_b_tt = {0: b_1_tt, 1: b_2_tt}
tt_block_amen(block_matrix_tt, block_b_tt, kickrank=2, verbose=True)
#res_tt = tt_sub(tt_matrix_vec_mul(s_matrix_tt, sol), b_tt)
#print(tt_inner_prod(res_tt, res_tt))


