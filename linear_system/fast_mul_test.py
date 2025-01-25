import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *


np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
matrix_tt = tt_random_gaussian([4, 4], shape=(2, 2))
vec_tt = tt_random_gaussian([4, 4], shape=(2,))
matrix_tt_2 = tt_random_gaussian([4, 4], shape=(2, 2))

print("Ground truth:")
print(tt_matrix_to_matrix(matrix_tt) @ tt_vec_to_vec(vec_tt))

print("Fast test:")
print(tt_vec_to_vec(tt_fast_matrix_vec_mul(matrix_tt, vec_tt)))


print("Ground truth:")
print(tt_matrix_to_matrix(matrix_tt) @ tt_matrix_to_matrix(matrix_tt_2))

print("Fast test:")
print(tt_matrix_to_matrix(tt_fast_mat_mat_mul(matrix_tt, matrix_tt_2)))


print("Ground truth:")
print(tt_matrix_to_matrix(matrix_tt) * tt_matrix_to_matrix(matrix_tt_2))

print("Fast test:")
print(tt_matrix_to_matrix(tt_fast_hadammard(matrix_tt, matrix_tt_2)))

print("Ground truth:")
print(tt_vec_to_vec(vec_tt) * tt_vec_to_vec(vec_tt))

print("Fast test:")
print(tt_vec_to_vec(tt_fast_hadammard(vec_tt, vec_tt)))

print(tt_matrix_to_matrix(matrix_tt))
print(tt_matrix_to_matrix(tt_rank_retraction(tt_diag(tt_diagonal(matrix_tt)), [1, 1])))
