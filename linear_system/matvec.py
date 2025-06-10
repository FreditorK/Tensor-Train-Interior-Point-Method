import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_als import tt_approx_mat_vec_mul


np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)

s_matrix_tt = tt_random_gaussian([4, 4], shape=(2, 2))
s_vec = tt_random_gaussian([4, 4], shape=(2,))

result_1 = tt_fast_matrix_vec_mul(s_matrix_tt, s_vec)

print(tt_vec_to_vec(result_1).flatten())

result_2 = tt_approx_mat_vec_mul(s_matrix_tt, s_vec)

print(tt_vec_to_vec(result_2).flatten())