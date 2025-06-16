import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
import time
import copy

np.random.seed(6)
np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
s_matrix_tt_01 = tt_rank_reduce(tt_scale(5, tt_random_gaussian([4, 10, 8, 8, 4], shape=(2, 2))))
s_matrix_tt_02 = tt_rank_reduce(tt_scale(-1, tt_random_gaussian([4, 5, 8, 2, 4], shape=(2, 2))))
s_matrix_tt_1 = tt_rank_reduce(tt_sum(s_matrix_tt_01, s_matrix_tt_02, tt_identity(7)))
print("Orig: ", tt_ranks(s_matrix_tt_1))

for i in range(10):
    s_matrix_tt_2 = tt_rank_retraction(s_matrix_tt_1, [8]*5)
    s_matrix_tt_1 = tt_rank_reduce(tt_sub(s_matrix_tt_1, s_matrix_tt_2))
    print(f"{i}: ", tt_ranks(s_matrix_tt_1))