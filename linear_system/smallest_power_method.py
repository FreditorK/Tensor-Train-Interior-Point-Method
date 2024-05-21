import sys
import os
import time

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_op import *

@dataclass
class Config:
    num_columns = 32
    tt_length = 8
    tt_max_rank = 18

np.random.seed(11)

#columns = [
#    tt_scale(10 * np.random.rand(), tt_random_binary([
#        np.random.randint(low=1, high=Config.tt_max_rank + 1) for _ in range(Config.tt_length - 1)
#    ])) for _ in range(Config.num_columns)
#]


#tensor_matrix, length = tt_tensor_matrix(columns)
#gram_tensor = tt_gram(tensor_matrix)
gram_tensor = tt_random_matrix([4, 5, 4])
print("--------")
t0 = time.time()
tt_eig, tt_eig_val = tt_randomised_min_eigentensor(gram_tensor, num_iter=25)
t1 = time.time()
#print("Eigen tensor: \n", tt_to_tensor(tt_eig))
print("Gram ranks: ", [c.shape for c in gram_tensor])
print("Ranks of eigen tensor: ", [c.shape for c in tt_eig])
print("Eigen value: ", tt_eig_val)
print(f"Power Method converged in {t1-t0:.4f}s")