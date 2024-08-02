import sys
import os
import time
from copy import copy

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_ops import *
from src.tt_ops import *


@dataclass
class Config:
    num_columns = 2
    tt_length = 6
    tt_max_rank = 5


np.random.seed(42)

"""
A = tt_random_gaussian([4, 3, 4], shape=(2, ))
B = tt_random_gaussian(tt_ranks(A), shape=(2, ))
diff = tt_sub(B, A)
print("Error: ", tt_inner_prod(diff, diff))
B = tt_als(A)
diff = tt_sub(B, A)
print("Error: ", tt_inner_prod(diff, diff))
"""

A = tt_random_gaussian([2, 3], shape=(2, 2))
A_psd = tt_linear_op_compose(A, tt_transpose(A))
B_psd = tt_burer_monteiro_factorisation(A_psd)
diff = tt_sub(A_psd, tt_linear_op_compose(B_psd, tt_transpose(B_psd)))
print("Final Error: ", tt_inner_prod(diff, diff))

"""

import sympy as sp

# Define the dimensions of the matrices
m, n, p, q = sp.symbols('m n p q', integer=True)

# Define the matrices
A = sp.MatrixSymbol('A', n*m, n*m)
X = sp.MatrixSymbol('X', n, n)
C = sp.MatrixSymbol('C', m, m)
# Define the matrix equation E(X) = ||AXB - C||_F^2
X_kron = sp.kronecker_product(C, X)
AX = A*X_kron
E = sp.matrices.trace(AX)

# Compute the derivative of E with respect to X
dE_dX = sp.diff(E, X)

# Simplify the derivative
dE_dX_simplified = sp.simplify(dE_dX)

print("Matrix equation:")
print(E)
print("\nDerivative of E with respect to X:")
print(dE_dX_simplified)
"""

"""
A_copy = copy(A)
B = tt_rank_reduce(A, tt_bound=1e-6)
A = tt_generalised_nystroem(A_copy, tt_ranks(A))
diff = tt_sub(B, A)
print(tt_inner_prod(diff, diff))
"""

"""
print(tt_inner_prod(z_sym, y_sym))
k = tt_linear_op(z_sym, y)
print(tt_inner_prod(y, k))
"""
