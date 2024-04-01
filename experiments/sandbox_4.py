import os
import sys

sys.path.append(os.getcwd() + '/../')
import numpy as np


A = 0.5*np.array([[-3, 0], [0, 1]])
B = 0.2*np.array([[1, 2], [4, -1]])
C = 0.5*np.array([[2, 0], [0, -0.1]])

print("Truth: ", np.sum(A @ B @ C))
l = 1

theta = (0.135)**(1/3)
A = np.block([[theta, np.zeros((1, A.shape[1]))], [np.zeros((A.shape[0], 1)), -A]])
B = np.block([[theta, np.zeros((1, B.shape[1]))], [np.zeros((B.shape[0], 1)), B]])
C = np.block([[theta, np.zeros((1, C.shape[1]))], [np.zeros((C.shape[0], 1)), C]])

a = 0
k = 1
for _ in range(k):
    r_1 = np.random.randn(3, l)
    r_2 = np.random.randn(3, l)
    r_3 = np.random.randn(3, l)
    r_4 = np.random.randn(3, l)
    AY = A @ r_1
    QAY, _ = np.linalg.qr(AY)
    Down_A = QAY.T @ A
    QA, RA = np.linalg.qr(Down_A)

    B_p = RA @ B
    BY = B_p @ r_2
    QBY, _ = np.linalg.qr(BY)
    Down_B = QBY.T @ B_p
    QB, RB = np.linalg.qr(Down_B)

    C_p = RB @ C
    CY = C_p @ r_3
    QCY, _ = np.linalg.qr(CY)
    Down_C = QCY.T @ C_p
    QC, RC = np.linalg.qr(Down_C)
    a += np.sum(RC @ r_4)


print("Approximation: ", a/k)

