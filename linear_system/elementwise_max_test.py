import sys
import os


sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_als import tt_divide
from src.regular_ipm import vec, mat
from src.tt_eig import tt_elementwise_max

np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
matrix_tt_1 = tt_random_gaussian([3, 2, 5], shape=(2, ))
print(tt_vec_to_vec(matrix_tt_1).flatten())
matrix_tt_1 = tt_elementwise_max(matrix_tt_1, 0)
print(tt_vec_to_vec(matrix_tt_1).flatten())
