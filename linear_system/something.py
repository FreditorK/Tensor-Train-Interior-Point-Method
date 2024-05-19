import sys
import os
import time
from copy import copy

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_op import *

@dataclass
class Config:
    num_columns = 2
    tt_length = 6
    tt_max_rank = 5

np.random.seed(5)

columns = [
    tt_scale(10 * np.random.rand(), tt_random_binary([
        np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)
    ])) for _ in range(Config.num_columns)
]
y = tt_scale(10 * np.random.rand(), tt_random_binary([
        np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)
    ]))
z = tt_scale(10 * np.random.rand(), tt_random_binary([
        np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)
    ]))

z_sym = [np.kron(np.expand_dims(c, 1), np.expand_dims(c, 2)) for c in z]

y_sym = [np.kron(np.expand_dims(c, 1), np.expand_dims(c, 2)) for c in y]

print(tt_inner_prod(z_sym, y_sym))
k = tt_linear_op(z_sym, y)
print(tt_inner_prod(y, k))
