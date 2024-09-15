import sys
import os
import time

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_ops import *
from src.tt_factorisation import tt_burer_monteiro_factorisation


@dataclass
class Config:
    num_columns = 2
    tt_length = 6
    tt_max_rank = 5


np.random.seed(482)

A = tt_random_gaussian([4, 4, 4], shape=(2, 2))
A_psd = tt_rank_reduce(tt_mat_mat_mul(A, tt_transpose(A)))
t0 = time.time()
B_psd, err = tt_burer_monteiro_factorisation(A_psd, num_swps=15, max_iter=10)
t1 = time.time()
diff = tt_sub(A_psd, tt_mat_mat_mul(B_psd, tt_transpose(B_psd)))
print("Final Error: ", tt_inner_prod(diff, diff), f"Time: {t1-t0}s")
