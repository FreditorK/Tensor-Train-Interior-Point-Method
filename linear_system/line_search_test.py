import sys
import os

import scipy



sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
from src.tt_ops import tt_rank_reduce
from src.tt_eigen import tt_max_generalised_eigen

np.random.seed(4) # 9

X_tt = tt_random_gaussian([2, 9, 3, 7, 6], shape=(2, 2))
X_tt = tt_fast_mat_mat_mul(tt_transpose(X_tt), X_tt)

s_matrix_tt = tt_random_gaussian([4, 2, 2, 3, 5], shape=(2, 2))
s_matrix_tt = tt_add(tt_rank_reduce(tt_add(s_matrix_tt, tt_transpose(s_matrix_tt)), eps=0), X_tt)


print("----")
A = tt_matrix_to_matrix(X_tt)
B = tt_matrix_to_matrix(s_matrix_tt)

#print(np.linalg.eigvals(tt_matrix_to_matrix(X_tt)))
#print(np.linalg.eigvals(tt_matrix_to_matrix(s_matrix_tt)))
L_inv = np.linalg.inv(scip.linalg.cholesky(A, check_finite=False, lower=True))
eig_val, _ = scip.sparse.linalg.eigsh(-L_inv @ B @ L_inv.T, k=1, which="LA")
step_size = 1 / eig_val[0]
print(step_size)
print(np.linalg.eigvals(A + step_size*B))

#print(tt_min_eig(tt_add(X_tt, s_matrix_tt))[0], np.linalg.eigvals(A + B))
alpha, a = tt_max_generalised_eigen(X_tt, s_matrix_tt, op_tol=1e-12, verbose=True)
print(alpha, a)
print(np.linalg.eigvals(A + alpha*B))
