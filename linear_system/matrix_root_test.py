import sys
import os


sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_eig import *


np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
matrix_tt_1 = tt_random_graph(4, 2)
print(tt_matrix_to_matrix(matrix_tt_1))
print(tt_ranks(matrix_tt_1))


