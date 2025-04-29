import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ineq_check import *

np.random.seed(4)
np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=6, suppress=True)

matrix_tt_1 = tt_scale(0.009, tt_add(tt_one_matrix(6), [np.array([[1, 0], [0, 0]]).reshape(1, 2, 2, 1)] + tt_scale(0.21, tt_identity(5))))
matrix_tt_1 = tt_diag(tt_split_bonds(matrix_tt_1))
a = tt_random_gaussian([1, 4, 1, 3], shape=(2, 2))
matrix_tt_2 = tt_diag(tt_split_bonds(tt_add(tt_scale(0.001, tt_one_matrix(6)), [np.array([[1, 0], [0, 0]]).reshape(1, 2, 2, 1)] + a)))

step_size, _ = tt_ineq_optimal_step_size(matrix_tt_1, matrix_tt_2, 1e-8, verbose=True, nswp=6)
A = np.diagonal(tt_matrix_to_matrix(matrix_tt_1))
B = np.diagonal(tt_matrix_to_matrix(matrix_tt_2))
print(step_size)
#print(A+step_size*B, np.min(A+step_size*B))
true_step_size = -A / B
true_step_size = min(np.min((true_step_size > 0)*true_step_size + (true_step_size <= 0)*1), 1)
print("----True")
print("True step size: ", true_step_size)
#print(A+true_step_size*B, np.min(A+true_step_size*B))
print(np.argmin(A+true_step_size*B))

