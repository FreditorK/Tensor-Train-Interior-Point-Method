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


tensor_matrix, length = tt_tensor_matrix(columns)
print([c.shape for c in tensor_matrix])
gram_tensor = tt_gram(tensor_matrix)
print([c.shape for c in gram_tensor])
matrix = np.array([[tt_inner_prod(columns[i], columns[j]) for i in range(len(columns))] for j in range(len(columns))])
print(scipy.linalg.eigvals(matrix))
matrix = matrix/np.sqrt(np.trace(matrix.T @ matrix))
print(scipy.linalg.eigvals(matrix))
matrix = 2*np.eye(4) - matrix
print(scipy.linalg.eigvals(matrix))
x = np.random.randn(4, 1)
for _ in range(30):
    x = matrix @ x
    x = x / np.linalg.norm(x)
y = matrix @ x
y = x/np.linalg.norm(y)
x = matrix @ y
print(x.T @ y)
tt_eig, tt_eig_val = tt_smallest_power_method(gram_tensor, num_iter=50)
print("Eigen tensor: \n", tt_to_tensor(tt_eig))
print("Ranks of eigen tensor: ", [c.shape for c in tt_eig])
print("Eigen value: ", tt_eig_val)