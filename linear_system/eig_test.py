import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_als import tt_min_eig
import time
import copy

np.random.seed(6)
np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
s_matrix_tt = tt_rank_reduce(tt_scale(5, tt_random_gaussian([4], shape=(2, 2))))
s_matrix_tt = tt_add(tt_transpose(s_matrix_tt), s_matrix_tt)

diag_S = tt_diag_op(s_matrix_tt)
print("Mine: ", tt_min_eig(s_matrix_tt, return_eig_val=True, verbose=True)[1])
S = tt_matrix_to_matrix(s_matrix_tt)
print("Actual min eig: ", scp.sparse.linalg.eigsh(S, k=1, which="SA")[0])


print("Mine: ", tt_min_eig(tt_diag_op(s_matrix_tt), return_eig_val=True)[1])
S = tt_matrix_to_matrix(s_matrix_tt).flatten()
print("Actual min eig: ", np.sum(scp.sparse.linalg.eigsh(np.diag(S), k=1, which="SA")[0]))
