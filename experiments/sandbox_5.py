import os
import sys
sys.path.append(os.getcwd() + '/../')
import numpy as np
from src.tt_op import _decomposed_matrix_product

a = 0.5 * np.array([[1, 2, 1]])
B = -0.8 * np.array([[2, 1, 3], [-2, -1, -5], [3, 2, 0.4]])
c = 0.8 * np.array([[1], [5.3], [2.1]])
a = a/ np.linalg.norm(a)
B = B / np.linalg.norm(B)
c = c / np.linalg.norm(c)
result = a @ B @ c
approximation = _decomposed_matrix_product([a, B, c], iterations=100, error_bound=1)
print("Actual result: ", result.item(), np.trace(B @ (c @ a)))
print(f"Decomposed approximation: {approximation}={np.prod(approximation)}")