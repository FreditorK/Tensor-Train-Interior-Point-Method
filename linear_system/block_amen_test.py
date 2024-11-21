import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.ops import *
from src.tt_ops import _tt_core_collapse, _tt_lr_random_orthogonalise, \
    tt_randomised_min_eigentensor, tt_rank_reduce, _tt_generalised_nystroem, _tt_mat_core_collapse, tt_matrix_vec_mul
from src.tt_amen import tt_amen, tt_block_amen

np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
np.random.seed(4)
ranks = [2]
block_1_tt = tt_rank_reduce(tt_random_gaussian([4, 2], shape=(2, 2)))
block_2_tt = tt_rank_reduce(tt_random_gaussian([2, 3], shape=(2, 2)))
block_matrix_tt = {(0, 0): block_1_tt, (1, 1): block_2_tt}

b_1_tt = tt_rank_reduce(tt_matrix_vec_mul(block_1_tt, tt_random_gaussian([1, 2], shape=(2,))))
b_2_tt = tt_rank_reduce(tt_matrix_vec_mul(block_2_tt, tt_random_gaussian([3, 1], shape=(2,))))
block_b_tt = {0: b_1_tt, 1: b_2_tt}
sol, _ = tt_block_amen(block_matrix_tt, block_b_tt, kickrank=2, verbose=True)
block_sol_1 = sol[:-1] + [sol[-1][:, 0]]
block_sol_2 = sol[:-1] + [sol[-1][:, 1]]

res_tt_1 = tt_sub(tt_matrix_vec_mul(block_1_tt, block_sol_1), b_1_tt)
print("Res block 1: ", tt_inner_prod(res_tt_1, res_tt_1))
res_tt_2 = tt_sub(tt_matrix_vec_mul(block_2_tt, block_sol_2), b_2_tt)
print("Res block 2: ", tt_inner_prod(res_tt_2, res_tt_2))


