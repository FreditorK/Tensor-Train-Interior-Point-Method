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

import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np

# Define a random PSD matrix
def generate_psd_matrix(n):
    A = np.random.randn(n, n)
    return A @ A.T

# Burer-Monteiro factorization function
def burer_monteiro_factorization(X, p, num_steps=100, learning_rate=0.01):
    n = X.shape[0]
    # Initialize Y with random values
    Y = jax.random.normal(jax.random.PRNGKey(0), (n, p))

    def loss(Y):
        YYT = jnp.dot(Y, Y.T)
        return jnp.sum((YYT - X) ** 2)

    loss_grad = grad(loss)

    # Optimization loop
    for _ in range(num_steps):
        grads = loss_grad(Y)
        Y -= learning_rate * grads

    return Y

# Generate a PSD matrix
n = 5
X = generate_psd_matrix(n)
print("Original PSD Matrix:\n", X)

# Perform Burer-Monteiro factorization
p = 5  # Desired lower rank
Y = burer_monteiro_factorization(X, p)
print("Factor Matrix Y:\n", Y)

# Reconstruct the PSD matrix
X_reconstructed = jnp.dot(Y, Y.T)
print("Reconstructed PSD Matrix:\n", X_reconstructed)
print("Error: ", np.sum((X - X_reconstructed)**2))
