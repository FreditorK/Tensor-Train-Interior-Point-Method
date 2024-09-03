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
A_copy = copy.copy(A)
print(tt_inner_prod(A, A_copy))
A_copy = core_forward_orthogonalise(1, A_copy)
print(tt_inner_prod(A, A_copy))