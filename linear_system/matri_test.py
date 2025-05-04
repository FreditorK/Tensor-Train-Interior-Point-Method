import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *


np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
s_matrix_tt = tt_random_gaussian([4, 4, 4, 5, 2], shape=(2, 2))


op_1 = tt_diag_op(s_matrix_tt)
op_2 = tt_reshape(tt_diag(tt_split_bonds(s_matrix_tt)), (4, 4))

res = tt_sub(op_1, op_2)

print(tt_inner_prod(res, res))