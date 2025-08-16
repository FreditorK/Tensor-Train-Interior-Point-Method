import time
import sys
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import block_diag

# --- Problem setup ---
np.random.seed(42)
n1, n2 = 60, 60
block_size = 10

# K_y block-diagonal
B1 = np.random.rand(10,10)
B2 = np.random.rand(10,10)
B3 = np.random.rand(10,10)
B4 = np.random.rand(10,10)
B5 = np.random.rand(10,10)
B6 = np.random.rand(10,10)
K_y_dense = block_diag(B1 + B1.T, B2 + B2.T, B3 + B3.T, B4 + B4.T, B5 + B5.T, B6 + B6.T)
K_y = sp.csr_matrix(K_y_dense)

# Z block-diagonal SPD
Z_dense = np.random.rand(n2, n2)
Z_dense = Z_dense + Z_dense.T + 1e-12*np.eye(n2)
Z = sp.csr_matrix(Z_dense)

# Off-diagonal blocks
L = sp.random(n1, n2, density=0.1, format='csr')
P = sp.random(n2, n1, density=0.1, format='csr')

# Full system
A = sp.bmat([[K_y, L],
             [P, Z]], format='csr')
b = np.random.rand(n1+n2)

# --- Precompute block solvers ---
def block_cholesky_solvers(block, block_size, reg=1e-12):
    """Returns a list of sparse Cholesky factorizations for SPD blocks"""
    n = block.shape[0]
    solvers = []
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        sub = block[start:end, start:end] + reg*sp.eye(end-start)
        # Use splu as simple sparse solver; could use sksparse.cholmod for better performance
        solvers.append(spla.splu(sub.tocsc()))
    return solvers

def block_ilu_solvers(block, block_size, drop_tol=1e-4, fill_factor=10):
    """Returns a list of ILU factorizations for each block"""
    n = block.shape[0]
    solvers = []
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        sub = block[start:end, start:end].tocsc()
        solvers.append(spla.spilu(sub, drop_tol=drop_tol, fill_factor=fill_factor))
    return solvers

def solve_block(solvers, x):
    """Apply block solver list to vector x"""
    n = len(x)
    y = np.zeros_like(x, dtype=np.float64)
    for i, solver in enumerate(solvers):
        start = i*block_size
        end = min(start+block_size, n)
        y[start:end] = solver.solve(x[start:end])
    return y

# --- Compute solvers ---
Z_solvers = block_cholesky_solvers(Z, block_size)
K_y_solvers = block_ilu_solvers(K_y, block_size)

# --- Triangular Schur preconditioner (upper-triangular) ---
def triangular_schur_preconditioner(x):
    x1, x2 = x[:n1], x[n1:]
    
    # Step 1: Solve Z block
    y2 = solve_block(Z_solvers, x2)
    
    # Step 2: Solve K_y block with correction from L*y2
    rhs1 = x1 - L @ y2
    y1 = solve_block(K_y_solvers, rhs1)
    
    return np.concatenate([y1, y2])

# --- LinearOperator for GMRES ---
M = spla.LinearOperator(A.shape, matvec=triangular_schur_preconditioner)

# --- GMRES with triangular Schur preconditioner ---
callback_counter = {'iter':0}
def cb(xk): callback_counter.update({'iter': callback_counter['iter']+1})
x_prec, info_prec = spla.lgmres(A, b, M=M, rtol=1e-8, inner_m=50, callback=cb)
iters_prec = callback_counter['iter']

# --- GMRES without preconditioner ---
callback_counter_no = {'iter':0}
def cb2(xk): callback_counter_no.update({'iter': callback_counter_no['iter']+1})
x_no, info_no = spla.lgmres(A, b, rtol=1e-8, inner_m=50, callback=cb2)
iters_no = callback_counter_no['iter']

# --- Results ---
print("GMRES without preconditioner iterations:", iters_no)
print("GMRES with triangular Schur block preconditioner iterations:", iters_prec)
print("Preconditioned solution error:", np.linalg.norm(A.dot(x_prec)-b))
print("Unpreconditioned solution error:", np.linalg.norm(A.dot(x_no)-b))






