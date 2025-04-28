import sys
import os


sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import _tt_fast_mat_mat_mul


np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
op_tt = tt_random_gaussian([4, 4, 4, 4, 4], shape=(2, 2))
vec_tt = tt_random_gaussian([4, 4], shape=(2,))
matrix_tt_2 = tt_random_gaussian([4, 4], shape=(2, 2))
print([c.shape for c in op_tt])
print([c.shape for c in matrix_tt_2])
print("Ground truth: ")
print(tt_matrix_to_matrix(tt_merge_bonds(tt_fast_matrix_vec_mul(op_tt, tt_split_bonds(matrix_tt_2)))))

op_tt = tt_merge_cores(op_tt)
print([c.shape for c in op_tt])
op_tt = tt_reshape(op_tt, (4, 4))
matrix_tt_2 = tt_reshape(matrix_tt_2, (4, ))
print([c.shape for c in op_tt])
print([c.shape for c in matrix_tt_2])

print("Mine: ")
print(tt_matrix_to_matrix(tt_reshape(tt_fast_matrix_vec_mul(op_tt, matrix_tt_2), (2, 2))))


