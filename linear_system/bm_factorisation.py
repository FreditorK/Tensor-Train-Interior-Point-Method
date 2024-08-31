import sys
import os
import time
from copy import copy

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_ops import *


@dataclass
class Config:
    num_columns = 2
    tt_length = 6
    tt_max_rank = 5


np.random.seed(482)

A = tt_random_gaussian([4, 5, 3], shape=(2, 2))
A_psd = tt_rank_reduce(tt_mat_mat_mul(A, tt_transpose(A)))
B_psd, err = tt_burer_monteiro_factorisation(A_psd)
diff = tt_sub(A_psd, tt_mat_mat_mul(B_psd, tt_transpose(B_psd)))
print("Final Error: ", tt_inner_prod(diff, diff))
