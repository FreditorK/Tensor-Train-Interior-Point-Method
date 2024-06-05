import sys
import os
import time
import matplotlib.pyplot as plt

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_ops import *

@dataclass
class Config:
    num_columns = 2
    tt_length = 6
    tt_max_rank = 5

np.random.seed(1813)

print("Creating Problem...")
column = tt_scale(np.random.randn(), tt_random_binary([np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)]))
A_sym_1 = [np.kron(np.expand_dims(c, 1), np.expand_dims(c, 2)) for c in column]
A_sym_1 = tt_normalise(A_sym_1)
column = tt_scale(np.random.randn(), tt_random_binary([np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)]))
A_sym_2 = [np.kron(np.expand_dims(c, 1), np.expand_dims(c, 2)) for c in column]
A_sym_2 = tt_normalise(A_sym_2)
column = tt_scale(np.random.randn(), tt_random_binary([np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)]))
A_sym_3 = [np.kron(np.expand_dims(c, 1), np.expand_dims(c, 2)) for c in column]
A_sym_3 = tt_normalise(A_sym_3)
column = tt_scale(np.random.randn(), tt_random_binary([np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)]))
A_sym_4 = [np.kron(np.expand_dims(c, 1), np.expand_dims(c, 2)) for c in column]
A_sym_4 = tt_normalise(A_sym_4)

test_X = tt_scale(5*np.random.rand() - 2.5, tt_random_binary([np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)]))
test_X = [np.kron(np.expand_dims(c, 1), np.expand_dims(c, 2)) for c in test_X]
test_X = tt_normalise(test_X)

constraints, index_length = tt_linear_op_from_columns([A_sym_1, A_sym_2, A_sym_3, A_sym_4])
bias = tt_rank_reduce(tt_eval_constraints(constraints, test_X))
constraints = tt_rank_reduce(constraints)

column = [np.abs(c*np.random.randn(*c.shape)) for c in tt_random_binary([np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)])]
Obj_sym = tt_rank_reduce([np.kron(np.expand_dims(c, 1), np.expand_dims(c, 2)) for c in column])
Obj_sym = Obj_sym

print("...Problem created!")
print(f"Objective Ranks: {tt_ranks(Obj_sym)}")
print(f"Constraint Ranks: As {tt_ranks(constraints)}, bias {tt_ranks(bias)}")
t0 = time.time()
X, duality_gaps = tt_sdp_fw(Obj_sym, constraints, bias, num_iter=100)
t1 = time.time()
print(f"Problem solved in {t1-t0}s")
print(f"Objective value: {tt_inner_prod(Obj_sym, X)}")
print(f"Avg Constraint error: {np.sum(np.abs(tt_to_tensor(tt_eval_constraints(constraints, X)) - tt_to_tensor(bias)))}")
print("Ranks of X: ", tt_ranks(X))
plt.plot(duality_gaps)
plt.show()