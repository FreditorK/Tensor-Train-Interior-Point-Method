import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.ops import *
from src.tt_ops import tt_rank_reduce
from src.tt_amen import tt_amen, tt_block_als

np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
np.random.seed(9)
ranks = [2]
block_00_tt = tt_rank_reduce(tt_random_gaussian([4, 2, 3, 4], shape=(2, 2)))
block_11_tt = tt_rank_reduce(tt_random_gaussian([2, 3, 2, 3], shape=(2, 2)))
block_21_tt = tt_rank_reduce(tt_random_gaussian([2, 3, 1, 3], shape=(2, 2)))
block_22_tt = tt_rank_reduce(tt_random_gaussian([2, 3, 4, 3], shape=(2, 2)))
block_00_tt = tt_fast_mat_mat_mul(tt_transpose(block_00_tt), block_00_tt)
block_11_tt = tt_fast_mat_mat_mul(tt_transpose(block_11_tt), block_11_tt)
block_21_tt = tt_fast_mat_mat_mul(tt_transpose(block_21_tt), block_21_tt)
block_22_tt = tt_fast_mat_mat_mul(tt_transpose(block_22_tt), block_22_tt)
block_matrix_tt = {(0, 0): block_00_tt, (1, 1): block_11_tt, (2, 1): block_21_tt, (2, 2): block_22_tt}
"""
[A_00           ]     [a_1]
[     A_11      ] x = [a_2]
[     A_21  A_22]     [a_3]
"""

b_1_tt = tt_rank_reduce(tt_fast_matrix_vec_mul(block_00_tt, tt_random_gaussian([1, 2, 3, 4], shape=(2,))))
b_2_tt = tt_rank_reduce(tt_fast_matrix_vec_mul(block_11_tt, tt_random_gaussian([3, 1, 3, 3], shape=(2,))))
temp_x = tt_random_gaussian([4, 3, 1, 2], shape=(2,))
b_3_tt = tt_rank_reduce(tt_add(tt_fast_matrix_vec_mul(block_21_tt, temp_x), tt_fast_matrix_vec_mul(block_22_tt, temp_x)))
block_b_tt = {0: b_1_tt, 1: b_2_tt, 2: b_3_tt}
sol, _ = tt_block_als(block_matrix_tt, block_b_tt, tol=1e-7, nswp=10, verbose=True)
block_sol_1 = sol[:-1] + [sol[-1][:, 0]]
block_sol_2 = sol[:-1] + [sol[-1][:, 1]]
block_sol_3 = sol[:-1] + [sol[-1][:, 2]]

res_tt_1 = tt_sub(tt_fast_matrix_vec_mul(block_00_tt, block_sol_1), b_1_tt)
print("Res block 1: ", tt_inner_prod(res_tt_1, res_tt_1))
res_tt_2 = tt_sub(tt_fast_matrix_vec_mul(block_11_tt, block_sol_2), b_2_tt)
print("Res block 2: ", tt_inner_prod(res_tt_2, res_tt_2))
res_tt_3 = tt_sub(tt_add(tt_fast_matrix_vec_mul(block_21_tt, block_sol_2), tt_fast_matrix_vec_mul(block_22_tt, block_sol_3)), b_3_tt)
print("Res block 2: ", tt_inner_prod(res_tt_3, res_tt_3))


