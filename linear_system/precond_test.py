import time
import sys
import os


sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import block_diag

# --- Problem setup ---
np.random.seed(42)
n1, n2 = 30, 30  # sizes of the main blocks
block_size = 15   # size of diagonal sub-blocks

# Example blocks
B1 = np.random.rand(10,10)
B2 = np.random.rand(10,10)
B3 = np.random.rand(10,10)
K_y_dense = block_diag(B1 + B1.T, B2 + B2.T, B3 + B3.T)
K_y = sp.csr_matrix(K_y_dense)

# Z: SPD block
Z_dense = np.random.rand(n2, n2)
Z_dense = Z_dense @ Z_dense.T + 1e-12*np.eye(n2)
Z = sp.csr_matrix(Z_dense)

# Off-diagonal blocks
L = sp.random(n1, n2, density=0.03, format='csr')
P = sp.random(n2, n1, density=0.03, format='csr')

# Full system
A = sp.bmat([[K_y, L],
             [P, Z]], format='csr')
b = np.random.rand(n1+n2)

# --- Helper: split a block into diagonal sub-blocks ---
def block_diagonal_inverse(block, block_size, reg=1e-8):
    n = block.shape[0]
    inv_blocks = []
    for start in range(0, n, block_size):
        end = min(start+block_size, n)
        sub = block[start:end, start:end].toarray() + reg*np.eye(end-start)
        inv_blocks.append(np.linalg.inv(sub))
    return inv_blocks

# Precompute inverses of diagonal sub-blocks
K_y_inv_blocks = block_diagonal_inverse(K_y, block_size)
Z_inv_blocks   = block_diagonal_inverse(Z, block_size)

# --- Block Jacobi preconditioner ---
def block_jacobi_preconditioner(x):
    x1, x2 = x[:n1], x[n1:]
    
    # Solve K_y blocks
    y1 = np.zeros_like(x1)
    for i, inv in enumerate(K_y_inv_blocks):
        start = i*block_size
        end = min(start+block_size, n1)
        y1[start:end] = inv @ x1[start:end]
    
    # Solve Z blocks
    y2 = np.zeros_like(x2)
    for i, inv in enumerate(Z_inv_blocks):
        start = i*block_size
        end = min(start+block_size, n2)
        y2[start:end] = inv @ x2[start:end]
    
    return np.concatenate([y1, y2])

M = spla.LinearOperator(A.shape, matvec=block_jacobi_preconditioner)

# --- GMRES with preconditioner ---
callback_counter = {'iter':0}
def cb(xk): callback_counter.update({'iter': callback_counter['iter']+1})
x_prec, info_prec = spla.lgmres(A, b, M=M, rtol=1e-8, inner_m=20, callback=cb)
iters_prec = callback_counter['iter']

# --- GMRES without preconditioner ---
callback_counter_no = {'iter':0}
def cb2(xk): callback_counter_no.update({'iter': callback_counter_no['iter']+1})
x_no, info_no = spla.lgmres(A, b, rtol=1e-8, inner_m=20, callback=cb2)
iters_no = callback_counter_no['iter']

# --- Results ---
print("GMRES without preconditioner iterations:", iters_no)
print("GMRES with block Jacobi iterations:", iters_prec)
print("Preconditioned solution error:", np.linalg.norm(A.dot(x_prec)-b))
print("Unpreconditioned solution error:", np.linalg.norm(A.dot(x_no)-b))





