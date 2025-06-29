import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import tt_rank_reduce
from src.tt_als import TTBlockMatrix, TTBlockVector, tt_block_amen
from src.tt_ipm import _tt_get_block

np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
np.random.seed(9)
ranks = [2]
block_00_tt = tt_rank_reduce(tt_random_gaussian([4, 2, 5, 8, 3, 4], shape=(2, 2)))
block_11_tt = tt_rank_reduce(tt_random_gaussian([2, 3, 9, 4, 2, 3], shape=(2, 2)))
block_21_tt = tt_rank_reduce(tt_random_gaussian([2, 3, 10, 6, 1, 7], shape=(2, 2)))
block_22_tt = tt_scale(3, tt_rank_reduce(tt_random_gaussian([2, 3, 9, 12, 4, 6], shape=(2, 2))))
block_00_tt = tt_add(tt_transpose(block_00_tt), block_00_tt)
block_11_tt = block_11_tt #tt_add(block_11_tt, tt_transpose(block_11_tt)) #tt_fast_mat_mat_mul(tt_transpose(block_11_tt), block_11_tt)
block_21_tt = tt_fast_mat_mat_mul(tt_transpose(block_21_tt), block_21_tt)
block_12_tt = tt_identity(len(block_21_tt))
block_22_tt = tt_fast_mat_mat_mul(tt_transpose(block_22_tt), block_22_tt)
block_matrix_tt = TTBlockMatrix()
block_matrix_tt[0, 0] = block_00_tt
block_matrix_tt[0, 1] = block_11_tt
block_matrix_tt.add_alias((0, 1), (1, 0), is_transpose=True)
block_matrix_tt[2, 1] = block_12_tt
block_matrix_tt[1, 2] = block_21_tt
#block_matrix_tt.add_alias((2, 1), (1, 2), is_transpose=True)
block_matrix_tt[2, 2] = block_22_tt
"""
[A_00   A_11      ]     [a_1]
[A_11.T       I   ] x = [a_2]
[       A_21  A_22]     [a_3]
"""

b_1_tt = tt_rank_reduce(tt_fast_matrix_vec_mul(block_00_tt, tt_random_gaussian([1, 6, 7, 2, 4, 4], shape=(2,))))
k_tt = tt_random_gaussian([3, 1, 9, 4, 3, 3], shape=(2,))
temp_x = tt_random_gaussian([4, 1, 4, 3, 2, 6], shape=(2,))
b_2_tt = tt_rank_reduce(tt_add(tt_fast_matrix_vec_mul(block_11_tt, k_tt), tt_fast_matrix_vec_mul(block_12_tt, temp_x)))
b_3_tt = tt_rank_reduce(tt_add(tt_fast_matrix_vec_mul(block_21_tt, k_tt), tt_fast_matrix_vec_mul(block_22_tt, temp_x)))
block_b_tt = TTBlockVector()
block_b_tt[0] = b_1_tt
block_b_tt[1] = b_2_tt
block_b_tt[2] = b_3_tt
print("b-ranks", tt_ranks(b_1_tt), tt_ranks(b_2_tt), tt_ranks(b_3_tt))
sol, _ = tt_block_amen(block_matrix_tt, block_b_tt, term_tol=1e-6, r_max=16, verbose=True, nswp=20, amen=True, kick_rank=2)
block_sol_1 = _tt_get_block(0, sol)
block_sol_2 =  _tt_get_block(1, sol)
block_sol_3 =  _tt_get_block(2, sol)

res_tt = block_matrix_tt.block_product(sol, 1e-8) - block_b_tt

print("Result: ", res_tt.norm)



