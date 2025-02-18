import sys
import os


sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_amen import tt_divide
from src.regular_ipm import vec, mat


np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
matrix_tt_1 = tt_random_gaussian([2], shape=(2, 2))
matrix_tt_1 = tt_fast_hadammard(matrix_tt_1, tt_sub(tt_one_matrix(len(matrix_tt_1)), tt_identity(len(matrix_tt_1))))
matrix_tt_1 = tt_rank_reduce(tt_add(matrix_tt_1, tt_transpose(matrix_tt_1)), 1e-8)

print(tt_matrix_to_matrix(matrix_tt_1))

def tt_diag_mat(matrix_tt):
    factor = [np.array([[1, 0], [0, 0]]).reshape(1, 2, 2, 1), np.array([[1, 0], [0, 0]]).reshape(1, 2, 2, 1)]
    print(tt_matrix_to_matrix(factor))
    basis =  tt_kron(factor, matrix_tt)
    return tt_rank_reduce(basis)

mat = tt_diag_mat(matrix_tt_1)
print(tt_matrix_to_matrix(mat))
