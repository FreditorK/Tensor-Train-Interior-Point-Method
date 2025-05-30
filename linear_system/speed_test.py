import time
import sys
import os


sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
import numpy as np


def idenity(dim):
    return [np.eye(2).reshape(1, 2, 2, 1) for _ in range(dim)]


t0 = time.time()
_ = idenity(100)
t1 = time.time()
_ = tt_identity(100)
t2 = time.time()

print(t1-t0, t2-t1)


X = tt_random_gaussian([3]*5, shape=(2, 2))
t0 = time.time()
X_a = tt_rank_reduce(X)
t1 = time.time()
X_b = tt_rank_reduce_py(X)
t2 = time.time()
diff = tt_sub(X_a, X)

print(tt_inner_prod(diff, diff))
print(t1-t0, t2-t1)



