import sys
import os

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_op import *

@dataclass
class Config:
    num_columns = 2
    tt_length = 4
    tt_max_rank = 5

np.random.seed(3)

column = tt_scale(2 * np.random.rand(), tt_random_binary([np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)]))
A_sym = [np.kron(np.expand_dims(c, 1), np.expand_dims(c, 2)) for c in column]
bias = 0.9

X = tt_sdp_frank_wolfe(A_sym, bias, num_iter=80)
print(f"AX={tt_inner_prod(A_sym, X)}, b={bias}")
print("Ranks of X: ", tt_ranks(X))


