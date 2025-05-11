import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_eigen import tt_approx_mat_mat_mul
import time

np.random.seed(6)
np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
s_matrix_tt = tt_random_gaussian([4, 10, 20, 10, 8], shape=(2, 2))
s_matrix_tt_2 = tt_random_gaussian([4, 10, 20, 10, 5], shape=(2, 2))

t0 = time.time()
res1 = tt_approx_mat_mat_mul(s_matrix_tt, s_matrix_tt_2, verbose=True)
t1 = time.time()
print()
res2 = tt_fast_mat_mat_mul(s_matrix_tt, s_matrix_tt_2)
t2 = time.time()

print(t1 -t0, t2-t1, tt_ranks(res1), tt_ranks(res2))