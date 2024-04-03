import os
import sys

sys.path.append(os.getcwd() + '/../')

from typing import List

import numpy as np
import jax.numpy as jnp
from itertools import product
from src.tt_op import *
import matplotlib.pyplot as plt


def sweep_to_lower_range():
    pass


A = 0.2 * np.array([[1, 2, 1], [-4, 1, -1]])
B = -0.08 * np.array([[2, 1], [-5, -1], [3, 2]])
C = 0.1 * np.array([[-1, 4], [5.3, 2]])
result = A @ B @ C
print("Actual result: ", np.sum(np.diagonal(result)))

# A = np.block([[-A, np.zeros((A.shape[0], 1))], [np.zeros((1, A.shape[1])), theta]])
# B = np.block([[B, np.zeros((B.shape[0], 1))], [np.zeros((1, B.shape[1])), theta]])
# C = np.block([[C, np.zeros((C.shape[0], 1))], [np.zeros((1, C.shape[1])), theta]])

m = 0
factors = []
K = 200
for _ in range(K):
    vec_1 = 2 * np.round(np.random.rand(A.shape[0], 1)) - 1
    vec_2 = 2 * np.round(np.random.rand(B.shape[0], 1)) - 1
    vec_3 = 2 * np.round(np.random.rand(C.shape[0], 1)) - 1
    a = (vec_1.T @ (A) @ vec_2).item()
    b = (vec_2.T @ (B) @ vec_3).item()
    c = (vec_3.T @ (C) @ vec_1).item()
    factors.append(np.array([a, b, c]))

theta = 1
factors = np.array(factors)
sign_count = K
iteration = 1
step = 0.5
while K / 2 - 1 > sign_count or sign_count > K / 2 + 1:
    sign_list = []
    for i in range(K):
        a = np.sign(theta - factors[i, 0])
        b = np.sign(theta + factors[i, 1])
        c = np.sign(theta + factors[i, 2])
        sign_list.append(np.array([a * b * c]))

    sign_list = np.array(sign_list)
    sign_count = len(sign_list[sign_list > 0])
    if sign_count > K / 2:
        print(f"Count {iteration}: {sign_count}, theta: {theta}")
        theta -= step
    elif sign_count < K / 2:
        print(f"Count {iteration}: {sign_count}, theta: {theta}")
        theta += step
    if step == 0:
        break
    iteration += 1
    step *= 0.5
print(f"Count: {sign_count}, theta: {theta}, theta^3: {theta ** 3}")

"""
G_1 = 2*np.round(np.random.rand(3, 1)) - 1
G_2 = 2*np.round(np.random.rand(2, 1)) - 1
G_3 = 2*np.round(np.random.rand(2, 1)) - 1

AY = A @ G_1
QAY, _ = np.linalg.qr(AY)
Down_A = QAY.T @ A
QA, RA = np.linalg.qr(Down_A)

BY = RA @ B @ G_2
QBY, _ = np.linalg.qr(BY)
Down_B = QBY.T @ RA @ B
QB, RB = np.linalg.qr(Down_B)

CY = RB @ C @ G_3
QCY, _ = np.linalg.qr(CY)
Down_C = QCY.T @ RB @ C

print(np.trace(QAY @ QA @ QBY @ QB @ QCY @ Down_C))
print(np.trace(QAY @ Down_C))
print(QAY)
print(Down_C)



# Create subplots
fig, axs = plt.subplots(4, 1, figsize=(8, 12))

# Loop through each row of the data and create a histogram for each
for i in range(3):
    # Create histogram
    axs[i].hist(factors[:, i], bins=500, color='skyblue', edgecolor='black')

    # Add labels and title
    axs[i].set_xlabel('Value')
    axs[i].set_ylabel('Frequency')
    axs[i].set_title(f'Histogram of Row {i+1}')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
"""
