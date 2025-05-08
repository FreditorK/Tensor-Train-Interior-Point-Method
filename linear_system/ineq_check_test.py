import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_eigen import *

np.random.seed(4)
np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=6, suppress=True)

matrix_tt_1 = tt_random_gaussian([6, 4, 5, 7, 3], shape=(2, 2))
matrix_tt_1 = tt_add(tt_scale(0.1, tt_one_matrix(6)), tt_fast_hadammard(matrix_tt_1, matrix_tt_1))
matrix_tt_1 = tt_diag(tt_split_bonds(matrix_tt_1))
matrix_tt_2 = tt_diag(tt_split_bonds(tt_add(tt_scale(0.05, tt_one_matrix(6)), tt_random_gaussian([6, 4, 5, 3, 4], shape=(2, 2)))))

step_size, _ = tt_max_generalised_eigen(matrix_tt_1, matrix_tt_2, 1e-12, verbose=True)
A = np.diagonal(tt_matrix_to_matrix(matrix_tt_1))
B = np.diagonal(tt_matrix_to_matrix(matrix_tt_2))
print(step_size)
#print(A+step_size*B, np.min(A+step_size*B))
true_step_size = -A/B
true_step_size = np.min(true_step_size[true_step_size > 0])
print("----True")
print("True step size: ", true_step_size)
#print(A+true_step_size*B, np.min(A+true_step_size*B))
print(np.argmin(A+true_step_size*B))

