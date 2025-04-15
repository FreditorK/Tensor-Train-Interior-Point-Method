import sys
import os


sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import _tt_fast_mat_mat_mul

np.random.seed(4)
np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
matrix_tt_1 = tt_random_gaussian([4, 4], shape=(2, 2))
matrix_tt_2 = tt_random_gaussian([4, 4], shape=(2, 2))
print(tt_matrix_to_matrix(matrix_tt_1) @ tt_matrix_to_matrix(matrix_tt_2))
print(tt_matrix_to_matrix(tt_fast_mat_mat_mul(matrix_tt_1, matrix_tt_2, eps=1e-18)))
print(tt_matrix_to_matrix(_tt_fast_mat_mat_mul(matrix_tt_1, matrix_tt_2, eps=1e-18)))


