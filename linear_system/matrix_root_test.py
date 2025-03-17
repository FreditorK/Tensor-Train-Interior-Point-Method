import sys
import os


sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_eig import *

np.random.seed(0)
np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
matrix_tt_1 = tt_random_gaussian([4, 4], (4, 4))
matrix_tt_2 = tt_random_gaussian([3, 5], (4, 4))

print(tt_inner_prod(matrix_tt_1, matrix_tt_2))


