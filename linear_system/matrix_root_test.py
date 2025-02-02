import sys
import os


sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_amen import tt_divide


np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
vec_tt_1 = tt_random_gaussian([2, 2], shape=(2, ))
vec_tt_2 = tt_random_gaussian([2, 2], shape=(2, ))

sol = tt_divide(vec_tt_1, vec_tt_2, degenerate=True)
print(np.divide(tt_vec_to_vec(vec_tt_1), tt_vec_to_vec(vec_tt_2)).flatten())
print(tt_vec_to_vec(sol).flatten())
print(tt_ranks(sol))

