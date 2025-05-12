import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_eigen import tt_approx_mat_mat_mul, tt_psd_rank_reduce
import time
import copy

np.random.seed(6)
np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
s_matrix_tt = tt_random_gaussian([4, 10, 8], shape=(2, 2))


print(np.min(np.linalg.eigvalsh(tt_matrix_to_matrix(s_matrix_tt))))
s_matrix_tt_psd = tt_psd_rank_reduce(copy.copy(s_matrix_tt), verbose=True)

print(np.min(np.linalg.eigvalsh(tt_matrix_to_matrix(s_matrix_tt_psd))))
print(np.linalg.norm(tt_matrix_to_matrix(s_matrix_tt_psd) - tt_matrix_to_matrix(s_matrix_tt)))