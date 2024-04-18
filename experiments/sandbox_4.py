import os
import sys

sys.path.append(os.getcwd() + '/../')
import numpy as np


A = np.array([[-0.3, 0], [-0.2, 1]])
B = np.array([[0.01, -0.72], [0.5, -0.1]])
C = np.array([[0.2, -0.05], [0.1, -0.1]])

print("Truth: ", np.trace(A @ B @ C))

Q_1, d_1, V_T_1 = np.linalg.svd(A)
Q_2, d_2, V_T_2 = np.linalg.svd(B)
Q_3, d_3, V_T_3 = np.linalg.svd(C)
print(d_1, d_2, d_3)
print(np.trace(np.diag(d_1) @ V_T_1 @ Q_2 @ np.diag(d_2) @ V_T_2 @ Q_3 @ np.diag(d_3) @ V_T_3 @ Q_1))
print(np.trace(np.diag(d_1) @ np.diag(d_2) @ np.diag(d_3)))







