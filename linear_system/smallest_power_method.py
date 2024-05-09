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
    tt_scale(10 * np.random.rand(), tt_random_binary([
        np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)
    ])) for _ in range(Config.num_columns)
]


tensor_matrix, length = tt_tensor_matrix(columns)
gram_tensor = tt_gram(tensor_matrix)
t0 = time.time()
tt_eig, tt_eig_val = tt_randomised_min_eigentensor(gram_tensor, num_iter=45)
t1 = time.time()
print("Eigen tensor: \n", tt_to_tensor(tt_eig))
print("Ranks of eigen tensor: ", [c.shape for c in tt_eig])
print("Eigen value: ", tt_eig_val)
print(f"Power Method converged in {t1-t0:.2f}s")