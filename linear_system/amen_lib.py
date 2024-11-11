import sys
import os

sys.path.append(os.getcwd() + '/../')

import time
import torch as tn
import torchtt as tntt
from src.tt_ops import *

A = tntt.random([(2,2),(2,2),(2,2)],[1,2,3,1])
x = tntt.random([2,2,2],[1,2,3,1])
b = A @ x

xs = tntt.solvers.amen_solve(A, b, x0=b, eps=1e-7, verbose=True)
print([c.shape for c in A.cores])

print(xs)
res = tt_sub(tt_matrix_vec_mul([c.numpy() for c in A.cores], [c.numpy() for c in xs.cores]), [c.numpy() for c in b.cores])
print(tt_inner_prod(res, res))
print('Relative residual error ', (A@xs-b).norm()/b.norm())
print('Relative error of the solution  ', (xs-x).norm()/x.norm())