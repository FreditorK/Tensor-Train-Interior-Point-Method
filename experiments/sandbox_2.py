import os
import sys

sys.path.append(os.getcwd() + '/../')

from typing import List

import numpy as np
import jax.numpy as jnp
from itertools import product
from src.tt_op import *

A = np.array([[1, 2, 1], [4, 1, -1]])
B = 0.00015*np.array([[2, 1], [-2, -1], [3, 2]])
C = 0.0002*np.array([[1, 1], [5.3, 2]])
result = A @ B @ C
print(result, np.sum(np.diagonal(result)))
AQ, AR = np.linalg.qr(A)
ABQ, ABR = np.linalg.qr(AR @ B)
ABCQ, ABCR = np.linalg.qr(ABR @ C)
ABQ = AQ @ ABQ
m = 0
factors = []
K = 10000
for _ in range(K):
    vec_1 = np.random.randn(ABQ.shape[0], 1)
    vec_2 = np.random.randn(ABCQ.shape[0], 1)
    vec_3 = np.random.randn(ABCR.shape[0], 1)
    #vec_4 = np.random.randn(1, 2)
    a = (vec_1.T @ ABQ @ vec_2).item()
    b = (vec_2.T @ ABCQ @ vec_3).item()
    c = (vec_3.T @ ABCR @ vec_1).item()
    factors.append(np.array([a, b, c]))
    m += a*b*c
print(m/K)
print(np.round(np.array(factors), decimals=2))
print(np.mean(np.array(factors), axis=0))
