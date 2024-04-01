import os
import sys

sys.path.append(os.getcwd() + '/../')
import numpy as np


A = np.array([[-0.3, 0], [-0.2, 1]])
B = np.array([[1, 0.2], [0.4, -0.1]])
C = np.array([[0.2, 0.5], [0, -0.1]])

print("Truth: ", np.trace(A @ B @ C))
l = 1

theta = (0.54)**(1/3)
A = np.block([[-A, np.zeros((A.shape[0], 1))], [np.zeros((1, A.shape[1])), theta]])
B = np.block([[B, np.zeros((B.shape[0], 1))], [np.zeros((1, B.shape[1])), theta]])
C = np.block([[C, np.zeros((C.shape[0], 1))], [np.zeros((1, C.shape[1])), theta]])

print(np.trace(A @ B @ C))
d = np.linalg.eigvals(A @ B @ C)
print(d)

