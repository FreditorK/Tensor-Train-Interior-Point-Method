import sys
import os

sys.path.append(os.getcwd() + '/../')

import numpy as np
import scipy as scip
import time
from src.tt_ops import *
from src.tt_eig import tt_min_eig, tt_max_eig
from cy_src.ops_cy import *



def tt_is_geq(linear_op_tt, X_tt, vec_b_tt, nswp=10, eps=1e-10, crit=1e-10, verbose=False):
    res_tt = tt_sub(tt_matrix_vec_mul(linear_op_tt, tt_vec(X_tt)), vec_b_tt)
    res_tt = tt_rl_orthogonalise(res_tt)
    A = tt_rank_reduce(tt_diag(res_tt))
    min_val, _, res = tt_min_eig(A, nswp=nswp, eps=eps, verbose=verbose)
    return np.greater_equal(min_val, -crit), min_val, res


def tt_is_geq_(X_tt, nswp=10, eps=1e-10, crit=1e-10, verbose=False):
    res_tt = tt_vec(X_tt)
    res_tt = tt_rl_orthogonalise(res_tt)
    A = tt_rank_reduce(tt_diag(res_tt))
    min_val, _, res = tt_min_eig(A, nswp=nswp, eps=eps, verbose=verbose)
    return np.greater_equal(min_val, -crit), min_val, res


def tt_is_leq(linear_op_tt, X_tt, vec_b_tt, nswp=10, eps=1e-10, crit=1e-10, verbose=False):
    res_tt = tt_sub(tt_matrix_vec_mul(linear_op_tt, tt_vec(X_tt)), vec_b_tt)
    res_tt = tt_rl_orthogonalise(res_tt)
    A = tt_rank_reduce(tt_diag(res_tt))
    max_val, _, res = tt_max_eig(A, nswp=nswp, eps=eps, verbose=verbose)
    return np.less_equal(max_val, crit), max_val, res
