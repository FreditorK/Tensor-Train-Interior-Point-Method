import sys
import os
import time

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_op import *

@dataclass
class Config:
    num_columns = 2
    tt_length = 6
    tt_max_rank = 5

np.random.seed(140)

print("Creating Problem...")
column = tt_scale(np.random.randn(), tt_random_binary([np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)]))
A_sym_1 = [np.kron(np.expand_dims(c, 1), np.expand_dims(c, 2)) for c in column]
A_sym_1 = tt_normalise(A_sym_1)
column = tt_scale(np.random.randn(), tt_random_binary([np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)]))
A_sym_2 = [np.kron(np.expand_dims(c, 1), np.expand_dims(c, 2)) for c in column]
A_sym_2 = tt_normalise(A_sym_2)

test_X = tt_scale(np.random.rand(), tt_random_binary([np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)]))
test_X = [np.kron(np.expand_dims(c, 1), np.expand_dims(c, 2)) for c in test_X]
test_X = tt_normalise(test_X)
b_1 = tt_inner_prod(A_sym_1, test_X)
b_2 = tt_inner_prod(A_sym_2, test_X)
bias = [np.array([b_1, b_2]).reshape(1, 2, 1)]

constraints, index_length = tt_tensor_matrix([A_sym_1, A_sym_2])

column = [5 * np.random.rand()*c for c in tt_random_binary([np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)])]
Obj_sym = [np.kron(np.expand_dims(c, 1), np.expand_dims(c, 2)) for c in column]
Obj_sym = tt_normalise(Obj_sym)

print("...Problem created!")
t0 = time.time()
X = tt_sdp_fw(Obj_sym, constraints, bias, bt=False, num_iter=100)
t1 = time.time()
print(f"Problem solved in {t1-t0}s")
print(f"Objective value: {tt_inner_prod(Obj_sym, X)}")
print(f"AX={tt_eval_constraints(constraints, X)}, \n b={bias}")
print("Ranks of X: ", tt_ranks(X))