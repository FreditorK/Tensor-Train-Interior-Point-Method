import sys
import os
sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_op import *

@dataclass
class Config:
    num_columns = 2
    tt_length = 10
    tt_max_rank = 2

np.random.seed(12)

columns = [
    tt_walsh_op_inv(tt_random_binary([
        np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)
    ])) for _ in range(Config.num_columns)
]

tensor_matrix, index_length = tt_tensor_matrix(columns)
# FIXME: rank reduction techniques should only be applied to the cores coming after the index cores
print("Ranks of tensor matrix: ", [c.shape for c in tensor_matrix])
tt_train_b = tt_walsh_op_inv(tt_random_binary([
    np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(index_length-1)
]))
print("Ranks of tensor bias: ", [c.shape for c in tt_train_b])
gram_tensor = tt_gram_tensor_matrix(tensor_matrix, index_length)
index_sol = tt_conjugate_gradient(gram_tensor, tt_train_b)
X_sol = tt_partial_inner_prod(index_sol, tensor_matrix)
print("Ranks of solution: ", [c.shape for c in X_sol])
print("Absolute error: ", np.abs(tt_inner_prod(tt_partial_inner_prod(tensor_matrix, X_sol, reversed=True), tt_train_b) - 1))

