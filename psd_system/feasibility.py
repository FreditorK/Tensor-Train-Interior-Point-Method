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

np.random.seed(7)

column = tt_scale(2.4 * np.random.rand(), tt_random_binary([np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)]))
A_sym = [np.kron(np.expand_dims(c, 1), np.expand_dims(c, 2)) for c in column]
A_sym = tt_normalise(A_sym)
bias = 0.84

column = [5 * np.random.randn()*c for c in tt_random_binary([np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)])]
Obj_sym = [np.kron(np.expand_dims(c, 1), np.expand_dims(c, 2)) for c in column]
Obj_sym = tt_normalise(Obj_sym)

X = tt_sdp_fw(Obj_sym, A_sym, bias, bt=True, num_iter=100)
print(f"Objective value: {tt_inner_prod(Obj_sym, X)}")
print(f"AX={tt_inner_prod(A_sym, X)}, b={bias}")
print("Ranks of X: ", tt_ranks(X))


