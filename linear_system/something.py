import sys
import os
import time
from copy import copy

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_op import *

@dataclass
class Config:
    num_columns = 2
    tt_length = 4
    tt_max_rank = 5

np.random.seed(9)

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
W = tt_sdp_add(y, z)
print([c.shape for c in W])
columns[0] = tt_swap_all(columns[0] + tt_swap_all(columns[0]))
columns[1] = tt_swap_all(columns[1] + tt_swap_all(columns[1]))
X = tt_add(columns[0], columns[1])
X = [core_bond(c_1, c_2) for c_1, c_2 in zip(X[:-1:2], X[1::2])]
#print([c.shape for c in tt_linear_op(x, tt_swap_all(w))])
#print(tt_inner_prod(I, w + tt_linear_op(x, tt_swap_all(w))))
W = W + tt_swap_all(W)
W = [core_bond(c_1, c_2) for c_1, c_2 in zip(W[:-1:2], W[1::2])]
print(tt_inner_prod(tt_linear_op(X, y), tt_linear_op(W, z)))
W_TX = tt_linear_op_compose(tt_transpose(W), X)
print(tt_inner_prod(tt_linear_op(W_TX, y), z))
X_TW = tt_linear_op_compose(tt_transpose(X), W)
print(tt_inner_prod(y, tt_linear_op(X_TW, z)))
