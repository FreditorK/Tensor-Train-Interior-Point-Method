import sys
import os

sys.path.append(os.getcwd() + '/../')

import time
import scikit_tt as scitt
from scikit_tt.solvers.sle import als as sota_als
from scikit_tt.solvers.sle import mals as sota_mals
from src.tt_ops import *
from tt_als import tt_amen


# TODO: You have to take gram matrix because the system may often be rank deficient
np.random.seed(1258)
L = tt_rank_reduce(tt_random_binary([3, 2, 4, 8, 3], shape=(4, 4)))
initial_guess = tt_scale(5, tt_random_gaussian([4, 3, 5, 6, 3], shape=(4,)))
B = tt_rank_reduce(tt_matrix_vec_mul(L, initial_guess))
L = tt_gram(L)
B = tt_rank_reduce(tt_matrix_vec_mul(tt_transpose(L), B))
#L = tt_add(L, [(1e-6)*np.eye(4).reshape(1, 4, 4, 1) for _ in range(len(L))])
initial_guess = tt_random_binary([r for r in tt_ranks(initial_guess)], shape=(4,))
t0 = time.time()
solution = tt_amen(L, initial_guess, B)
t1 = time.time()
res = tt_sub(tt_matrix_vec_mul(L, solution), B)
print(f"Time taken: {t1 - t0}s")
print("Error: ", tt_inner_prod(res, res))
"""
np.random.seed(1598)
L = tt_rank_reduce(tt_random_gaussian([3, 2, 4, 8, 3], shape=(4, 4)))
initial_guess = tt_random_gaussian([4, 3, 5, 6, 3], shape=(4,))
B = tt_rank_reduce(tt_matrix_vec_mul(L, initial_guess))
#L = tt_add(L, [(1e-6)*np.array([[1, 0], [0, 1]]).reshape(1, 2, 2, 1) for _ in range(len(L))])
B = [c.reshape(c.shape[0], c.shape[1], 1, c.shape[-1]) for c in B]
initial_guess = tt_random_gaussian([r+3 for r in tt_ranks(B)], shape=(4, 1))
L = scitt.TT(L)
B = scitt.TT(B)
initial_guess = scitt.TT(initial_guess)
t0 = time.time()
solution = sota_als(L, initial_guess, B, repeats=5)
t1 = time.time()
res = L.dot(solution) - B
res_other = tt_sub(tt_matrix_vec_mul(L.cores, [c.reshape(c.shape[0], c.shape[1], c.shape[-1]) for c in solution.cores]), [c.reshape(c.shape[0], c.shape[1], c.shape[-1]) for c in B.cores])
print(f"Time taken: {t1 - t0}s")
print("Error: ", res.norm(2)**2, tt_inner_prod(res_other, res_other))#, res_al.T @ res_al)
"""