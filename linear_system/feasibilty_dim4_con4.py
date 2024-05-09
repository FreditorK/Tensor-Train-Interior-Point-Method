import sys
import os
import time

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_op import *

@dataclass
class Config:
    num_columns = 4
    tt_length = 4
    tt_max_rank = 5

np.random.seed(12)

columns = [
    tt_scale(5*np.random.rand(), tt_random_binary([
        np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)
    ])) for _ in range(Config.num_columns)
]

tensor_matrix, index_length = tt_tensor_matrix(columns)
t0 = time.time()
gram_tensor = tt_gram(tensor_matrix)  # row_index + inner results
t1 = time.time()
tt_train_b = tt_linear_op(gram_tensor, tt_scale(3*np.random.rand(), tt_random_binary([
     np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(index_length-1)
])))
t2 = time.time()
index_sol = tt_conjugate_gradient(gram_tensor, tt_train_b)
t3 = time.time()
print("Ranks of tensor matrix: ", [c.shape for c in tensor_matrix])
print("Ranks of tensor bias: ", [c.shape for c in tt_train_b])
print("Ranks of index: ", [c.shape for c in index_sol])
X_sol = tt_linear_op(tt_transpose(tensor_matrix), index_sol)
print("Ranks of solution: ", [c.shape for c in X_sol])
residual = tt_add(tt_linear_op(tensor_matrix, X_sol), tt_scale(-1, tt_train_b))
print("Absolute error: ", np.abs(tt_inner_prod(residual, residual)))
print(f"Gram tensor computed in {t1-t0:.2f}s")
print(f"CG converged in {t3-t2:.2f}s")