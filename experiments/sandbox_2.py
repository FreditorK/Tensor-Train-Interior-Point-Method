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

A = 0.2*np.array([[1, 2, 1], [-4, 1, -1]])
B = 0.08*np.array([[2, 1], [-5, -1], [3, 2]])
C = 0.1*np.array([[-1, 4], [5.3, 2]])
result = A @ B @ C
print("Actual result: ", np.sum(np.diagonal(result)))

"""
m = 0
factors = []
K = 100
for _ in range(K):
    vec_1 = 2*np.round(np.random.rand(A.shape[0], 1))-1
    vec_2 = 2*np.round(np.random.rand(B.shape[0], 1))-1
    vec_3 = 2*np.round(np.random.rand(C.shape[0], 1))-1
    # TODO: We could rescale the random vectors through the sampling to equalise the factors
    #vec_4 = np.random.randn(1, 2)
    a = (vec_1.T @ (A) @ vec_2).item()
    b = (vec_2.T @ (B) @ vec_3).item()
    c = (vec_3.T @ (C) @ vec_1).item()
    factors.append(np.array([a, b, c]))
    m += a*b*c

# Rademacher is a better estimator
print("Approximated result: ", m/K)
factors = np.array(factors)
factor_sgns = np.prod(np.sign(factors), axis=1)
abs_factors = np.abs(factors)

norm = K**(1/3)
A = np.diag(factors[:, 0])/norm
B = np.diag(factors[:, 1])/norm
C = np.diag(factors[:, 2])/norm
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
print(QBY)
print(QCY)
print(Down_C)


"""

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

axs[3].hist(prod_factors, bins=500, color='skyblue', edgecolor='black')

# Add labels and title
axs[3].set_xlabel('Value')
axs[3].set_ylabel('Frequency')
axs[3].set_title(f'Prod Factors')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
"""
