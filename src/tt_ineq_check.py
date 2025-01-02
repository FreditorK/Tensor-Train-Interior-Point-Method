import sys
import os

sys.path.append(os.getcwd() + '/../')

from src.tt_ops import *
from src.tt_eig import tt_min_eig, tt_max_eig
from cy_src.ops_cy import *



def tt_is_geq(linear_op_tt, X_tt, vec_b_tt, nswp=10, eps=1e-10, crit=1e-10, verbose=False):
    res_tt = tt_sub(vec_b_tt, tt_matrix_vec_mul(linear_op_tt, tt_vec(X_tt)))
    norm = np.sqrt(tt_inner_prod(res_tt, res_tt))
    if norm > crit:
        res_tt = tt_scale(np.divide(1, norm), res_tt)
        A = tt_rank_reduce(tt_diag(res_tt), 0.5*eps)
        min_val, _, res = tt_min_eig(A, nswp=nswp, eps=eps, verbose=verbose)
        return np.greater(norm*min_val, -crit), norm*min_val, res
    return True, 0.0, 0.0


def tt_is_geq_(X_tt, nswp=10, eps=1e-10, crit=1e-10, verbose=False):
    res_tt = tt_vec(X_tt)
    norm = np.sqrt(tt_inner_prod(res_tt, res_tt))
    if norm > crit:
        res_tt = tt_scale(np.divide(1, norm), res_tt)
        A = tt_rank_reduce(tt_diag(res_tt), 0.5*eps)
        min_val, _, res = tt_min_eig(A, nswp=nswp, eps=eps, verbose=verbose)
        return np.greater(norm*min_val, -crit), norm*min_val, res
    return True, 0.0, 0.0


def tt_is_leq(linear_op_tt, X_tt, vec_b_tt, nswp=10, eps=1e-10, crit=1e-10, verbose=False):
    res_tt = tt_sub(vec_b_tt, tt_matrix_vec_mul(linear_op_tt, tt_vec(X_tt)))
    norm = np.sqrt(tt_inner_prod(res_tt, res_tt))
    if norm > crit:
        res_tt = tt_scale(np.divide(1, norm), res_tt)
        A = tt_rank_reduce(tt_diag(res_tt), 0.1*eps)
        max_val, _, res = tt_max_eig(A, nswp=nswp, eps=eps, verbose=verbose)
        return np.less(norm*max_val, crit), norm*max_val, res
    return True, 0.0, 0.0
