import sys
import os

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_op import *
import scipy

@dataclass
class Config:
    num_columns = 4
    tt_length = 4
    tt_max_rank = 5

np.random.seed(12)

columns = [
    tt_mul_scal(10*np.random.rand(), tt_random_binary([
        np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)
    ])) for _ in range(Config.num_columns)
]

tensor_matrix = tt_tensor_matrix_2(columns)
print([c.shape for c in tensor_matrix])
gram = tt_gram(tensor_matrix)
print([c.shape for c in gram])

"""
tensor_matrix_t = tt_matrix_transpose(tensor_matrix, index_length)
gram_tensor = tt_gram_tensor_matrix(tensor_matrix, tensor_matrix_t, index_length)
matrix = tt_to_tensor(gram_tensor).reshape(4, 4)
print(scipy.linalg.eigvals(matrix))
tt_eig, tt_eig_val = tt_smallest_power_method(gram_tensor, num_iter=1)
print("Eigen tensor: \n", tt_to_tensor(tt_eig))
print("Ranks of eigen tensor: ", [c.shape for c in tt_eig])
print("Eigen value: ", tt_eig_val)
"""