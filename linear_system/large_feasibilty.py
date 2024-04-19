import sys
import os

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import asdict, dataclass
from src.tt_op import *
from src.tt_op import _tt_core_collapse

@dataclass
class Config:
    num_columns = 2
    tt_length = 2
    tt_max_rank = 2

np.random.seed(12)

columns = [
    tt_walsh_op_inv(tt_random_binary([np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)])) for _
    in
    range(Config.num_columns)]

tensor_matrix, index_length = tt_tensor_matrix(columns)
tt_train_b = tt_walsh_op_inv(tt_random_binary([np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(index_length-1)]))
print(tt_train_b)
gram_tensor = tt_gram_tensor_matrix(tensor_matrix, index_length)
index_sol = tt_conjugate_gradient(gram_tensor, tt_train_b, num_iter=2)
print("Index:", index_sol)
gram_matrix = tt_to_tensor(gram_tensor)
print(np.linalg.inv(gram_matrix) @ tt_train_b[0].reshape(-1, 1))
print(gram_matrix @ index_sol)
print(tt_partial_inner_prod(gram_tensor, index_sol, reversed=True))

def tt_index(tt_index, tt_tensor):
    if len(tt_index) > 1:
        first_matrix = np.linalg.multi_dot([_tt_core_collapse(core_index, core) for core_index, core in zip(tt_index, tt_tensor[:len(tt_index)])])
    else:
        first_matrix = _tt_core_collapse(tt_index[0], tt_tensor[0])
    first_core = np.einsum("ab, bcd -> acd", first_matrix, tt_tensor[len(tt_index)])
    return [first_core] + tt_tensor[len(tt_index)+1:]

X_sol = tt_index(index_sol, tensor_matrix)
#print(tt_to_tensor(X_sol))
print(tt_partial_inner_prod(tensor_matrix, X_sol, reversed=True))

