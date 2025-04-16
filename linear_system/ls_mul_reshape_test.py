import sys
import os


sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *

np.random.seed(4)
np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
matrix_tt_1 = tt_random_gaussian([4, 4, 6], shape=(2, 2))
matrix_tt_2 = tt_random_gaussian([4, 4, 3], shape=(2, 2))
print(tt_matrix_to_matrix(matrix_tt_1) @ tt_matrix_to_matrix(matrix_tt_2))
mat = tt_fast_mat_mat_mul(matrix_tt_1, matrix_tt_2, eps=1e-8)
print(tt_matrix_to_matrix(mat))
print(tt_ranks(mat))
print(tt_ranks(tt_rank_reduce(mat, 1e-8)))

