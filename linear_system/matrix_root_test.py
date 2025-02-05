import sys
import os


sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_amen import tt_divide
from src.regular_ipm import vec, mat


np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
matrix_tt_1 = tt_random_gaussian([2, 2], shape=(2, 2))
matrix_tt_2 = tt_random_gaussian([2, 2], shape=(2, 2))
matrix =  tt_kron(tt_identity(len(matrix_tt_1)), matrix_tt_1)
print(tt_matrix_to_matrix(tt_mat(tt_matrix_vec_mul(matrix, tt_vec(matrix_tt_2)))))
print(mat(tt_matrix_to_matrix(matrix) @ vec(tt_matrix_to_matrix(matrix_tt_2))))