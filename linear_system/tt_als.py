import sys
import os

from scipy.constants import micro

sys.path.append(os.getcwd() + '/../')

import numpy as np
import scipy.linalg as lin
import copy
import time
from typing import List
from src.tt_ops import *
import scikit_tt as scitt
from scikit_tt.solvers.sle import als as sota_als
from src.tt_ops import tt_rank_reduce


def als(lhs, initial_guess, rhs, tol=1e-5):
    # define solution tensor
    solution = copy.copy(initial_guess)
    solution_ranks = [1] + tt_ranks(solution) + [1]
    solution_shape = [c.shape[1:-1] for c in solution]
    op_order = len(lhs)

    # define stacks
    stack_left_op = [None] * op_order
    stack_left_rhs = [None] * op_order
    stack_right_op = [None] * op_order
    stack_right_rhs = [None] * op_order

    for i in range(op_order - 1, -1, -1):
        __construct_stack_right_op(i, stack_right_op, lhs, solution)
        __construct_stack_right_rhs(i, stack_right_rhs, rhs, solution)

    res = tt_sub(tt_matrix_vec_mul(lhs, solution), rhs)
    err = tt_inner_prod(res, res)
    print("Error 0: ", err)
    it = 0

    while np.less_equal(tol, err) and it < 10:

        for i in range(op_order):
            __construct_stack_left_op(i, stack_left_op, lhs, solution)
            __construct_stack_left_rhs(i, stack_left_rhs, rhs, solution)

            if i < op_order - 1:
                micro_op = __construct_micro_matrix_als(i, stack_left_op, stack_right_op, lhs, solution_ranks)
                micro_rhs = __construct_micro_rhs_als(i, stack_left_rhs, stack_right_rhs, rhs,
                                                      solution_ranks)
                __update_core_als(i, micro_op, micro_rhs, solution, 'forward', solution_ranks, solution_shape)

        for i in range(op_order - 1, -1, -1):
            __construct_stack_right_op(i, stack_right_op, lhs, solution)
            __construct_stack_right_rhs(i, stack_right_rhs, rhs, solution)
            micro_op = __construct_micro_matrix_als(i, stack_left_op, stack_right_op, lhs, solution_ranks)
            micro_rhs = __construct_micro_rhs_als(i, stack_left_rhs, stack_right_rhs, rhs, solution_ranks)
            __update_core_als(i, micro_op, micro_rhs, solution, 'backward', solution_ranks, solution_shape)

        res = tt_sub(tt_matrix_vec_mul(lhs, solution), rhs)
        err = tt_inner_prod(res, res)
        it += 1
        print(f"Error {it}: ", err)

    return solution


def __construct_stack_left_op(i: int, stack_left_op: List[np.ndarray], operator, solution):
    if i == 0:
        stack_left_op[i] = np.array([1], ndmin=3)
    else:
        stack_left_op[i] = np.tensordot(stack_left_op[i - 1], solution[i - 1], axes=(0, 0))
        stack_left_op[i] = np.tensordot(stack_left_op[i], operator[i - 1], axes=([0, 2], [0, 2]))
        stack_left_op[i] = np.tensordot(stack_left_op[i], np.conj(solution[i - 1]),
                                        axes=([0, 2], [0, 1]))


def __construct_stack_left_rhs(i, stack_left_rhs, right_hand_side, solution):
    if i == 0:
        stack_left_rhs[i] = np.array([1], ndmin=2)
    else:
        stack_left_rhs[i] = np.tensordot(stack_left_rhs[i - 1], right_hand_side[i - 1], axes=(0, 0))
        stack_left_rhs[i] = np.tensordot(stack_left_rhs[i], np.conj(solution[i - 1]),
                                         axes=([0, 1], [0, 1]))


def __construct_stack_right_op(i: int, stack_right_op: List[np.ndarray], operator, solution):
    if i == len(operator) - 1:
        stack_right_op[i] = np.array([1], ndmin=3)
    else:
        stack_right_op[i] = np.tensordot(np.conj(solution[i + 1]), stack_right_op[i + 1], axes=(2, 2))
        stack_right_op[i] = np.tensordot(operator[i + 1], stack_right_op[i], axes=([1, 3], [1, 3]))
        stack_right_op[i] = np.tensordot(solution[i + 1], stack_right_op[i], axes=([1, 2], [1, 3]))


def __construct_stack_right_rhs(i, stack_right_rhs, right_hand_side, solution):
    if i == len(right_hand_side) - 1:
        stack_right_rhs[i] = np.array([1], ndmin=2)
    else:
        stack_right_rhs[i] = np.tensordot(np.conj(solution[i + 1]), stack_right_rhs[i + 1], axes=(2, 1))
        stack_right_rhs[i] = np.tensordot(right_hand_side[i + 1], stack_right_rhs[i], axes=([1, 2], [1, 2]))


def __construct_micro_matrix_als(i: int,
                                 stack_left_op: List[np.ndarray],
                                 stack_right_op: List[np.ndarray],
                                 operator, solution_ranks) -> np.ndarray:
    micro_op = np.tensordot(stack_left_op[i], operator[i], axes=(1, 0))
    micro_op = np.tensordot(micro_op, stack_right_op[i], axes=(4, 1))
    micro_op = micro_op.transpose([1, 2, 5, 0, 3, 4]).reshape(
        solution_ranks[i] * operator[i].shape[1] * solution_ranks[i + 1],
        solution_ranks[i] * operator[i].shape[2] * solution_ranks[i + 1])

    return micro_op


def __construct_micro_rhs_als(i: int,
                              stack_left_rhs: List[np.ndarray],
                              stack_right_rhs: List[np.ndarray],
                              right_hand_side, solution_ranks) -> np.ndarray:
    micro_rhs = np.tensordot(stack_left_rhs[i], right_hand_side[i], axes=(0, 0))
    micro_rhs = np.tensordot(micro_rhs, stack_right_rhs[i], axes=(2, 0))
    micro_rhs = micro_rhs.reshape(solution_ranks[i] * right_hand_side[i].shape[1] * solution_ranks[i + 1], 1)

    return micro_rhs


def __update_core_als(i: int,
                      micro_op: np.ndarray, micro_rhs: np.ndarray,
                      solution, direction: str, solution_ranks, solution_shape):
    solution[i], _, _, _ = np.linalg.lstsq(micro_op, micro_rhs, rcond=None)
    if direction == 'forward':
        [q, _] = lin.qr(
            solution[i].reshape(solution_ranks[i] * solution_shape[i][0], solution_ranks[i + 1]),
            overwrite_a=True, mode='economic', check_finite=False)
        solution_ranks[i + 1] = q.shape[1]
        solution[i] = q.reshape(solution_ranks[i], solution_shape[i][0], solution_ranks[i + 1])
    if direction == 'backward':
        if i > 0:
            [_, q] = lin.rq(
                solution[i].reshape(solution_ranks[i], solution_shape[i][0] * solution_ranks[i + 1]),
                overwrite_a=True, mode='economic', check_finite=False)
            solution_ranks[i] = q.shape[0]
            solution[i] = q.reshape(solution_ranks[i], solution_shape[i][0], solution_ranks[i + 1])

        else:
            solution[i] = solution[i].reshape(solution_ranks[i], solution_shape[i][0], solution_ranks[i + 1])


def tt_infeasible_newton_system_rhs(obj_tt, linear_op_tt, bias_tt, X_tt, Y_tt, Z_tt, mu):
    idx_0 = [np.array([[1, 0], [0, 0]]).reshape(1, 2, 2, 1)]
    idx_1 = [np.array([[0, 1], [0, 0]]).reshape(1, 2, 2, 1)]
    idx_2 = [np.array([[0, 0], [1, 0]]).reshape(1, 2, 2, 1)]
    upper_rhs = idx_0 + tt_sub(tt_sub(tt_mat(tt_linear_op(tt_adjoint(linear_op_tt), Y_tt), shape=(2, 2)), Z_tt), obj_tt)
    middle_rhs = idx_1 + tt_sub(tt_mat(tt_linear_op(linear_op_tt, X_tt), shape=(2, 2)), bias_tt)
    lower_rhs = idx_2 + tt_sub(tt_scale(mu, tt_identity(len(X_tt))), tt_mat_mat_mul(Z_tt, X_tt))
    newton_rhs = tt_add(upper_rhs, middle_rhs)
    newton_rhs = tt_add(newton_rhs, lower_rhs)
    return tt_rank_reduce(tt_vec(newton_rhs))


def tt_infeasible_newton_system_lhs(linear_op_tt, X_tt, Z_tt):
    idx_01 = [np.einsum("i, j -> ij", np.array([1, 0, 1, 0]), np.array([1, 0, 0, 1])).reshape(1, 4, 4, 1)]
    idx_10 = [np.einsum("i, j -> ij", np.array([1, 0, 1, 0]), np.array([0, 1, 1, 0])).reshape(1, 4, 4, 1)]
    idx_02 = [np.einsum("i, j -> ij", np.array([1, 0, 0, 1]), np.array([1, 0, 1, 0])).reshape(1, 4, 4, 1)]
    idx_20 = [np.einsum("i, j -> ij", np.array([0, 1, 1, 0]), np.array([1, 0, 1, 0])).reshape(1, 4, 4, 1)]
    idx_22 = [np.einsum("i, j -> ij", np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0])).reshape(1, 4, 4, 1)]
    linear_op_tt = tt_scale(-1, linear_op_tt)
    linear_op_tt_adjoint = idx_01 + tt_op_to_mat(tt_adjoint(linear_op_tt))
    linear_op_tt = tt_op_to_mat(linear_op_tt)
    I_op_tt = idx_02 + tt_op_to_mat(tt_op_from_tt_matrix(tt_identity(len(X_tt))))
    Z_op_tt = idx_20 + tt_op_to_mat(tt_op_from_tt_matrix(Z_tt))
    X_op_tt = idx_22 + tt_op_to_mat(tt_op_from_tt_matrix(X_tt))
    newton_system = tt_add(idx_10 + linear_op_tt, linear_op_tt_adjoint)
    newton_system = tt_add(newton_system, I_op_tt)
    newton_system = tt_add(newton_system, Z_op_tt)
    newton_system = tt_add(newton_system, X_op_tt)
    # TODO: Preconditioning
    return tt_rank_reduce(newton_system)


def _tt_get_block(i, j, block_matrix_tt):
    first_core = block_matrix_tt[0][:, i, j, :]
    first_core = np.einsum("ab, bcde -> acde", first_core, block_matrix_tt[1])
    return [first_core] + block_matrix_tt[2:]


def _tt_ipm_newton_step(obj_tt, linear_op_tt, bias_tt, XZ_tt, Y_tt):
    X_tt = _tt_get_block(0, 0, XZ_tt)
    Z_tt = _tt_get_block(0, 1, XZ_tt)
    lhs_matrix_tt = tt_infeasible_newton_system_lhs(linear_op_tt, X_tt, Z_tt)
    mu = tt_inner_prod(Z_tt, [0.5 * c for c in X_tt])
    rhs_vec_tt = tt_infeasible_newton_system_rhs(obj_tt, linear_op_tt, bias_tt, X_tt, Y_tt, Z_tt, mu)
    np.set_printoptions(threshold=np.inf, linewidth=200)
    K = np.round(tt_matrix_to_matrix(lhs_matrix_tt), decimals=2)
    K_inv = np.linalg.pinv(K)
    # ----
    initial_guess = tt_random_gaussian(tt_ranks(rhs_vec_tt), shape=(4,))
    Delta_tt = als(lhs_matrix_tt, initial_guess, rhs_vec_tt)
    #return Delta_tt


def _tt_homotopy_step(XZ_tt, Delta_XZ_tt):
    XZ_tt = tt_rank_reduce(tt_add(XZ_tt, Delta_XZ_tt))
    # Projection to PSD-cone
    V_tt = tt_burer_monteiro_factorisation(XZ_tt)
    return V_tt


def _symmetrisation(Delta_tt):
    return tt_rank_reduce(tt_scale(0.5, tt_add(Delta_tt, tt_transpose(Delta_tt))))


def _get_xz_block(XYZ_tt):
    new_index_block = np.zeros_like(XYZ_tt[0])
    new_index_block[:, 0] += XYZ_tt[0][:, 0]
    return new_index_block + XYZ_tt[1:]


def tt_ipm(obj_tt, linear_op_tt, bias_tt):
    dim = len(obj_tt)
    V_tt = [np.array([[1, 0], [1, 0]]).reshape(1, 2, 2, 1)] + tt_identity(dim)  # [X, 0, Z, 0]^T
    Y_tt = tt_zeros(dim, shape=(2, 2))
    for _ in range(1):
        XZ_tt = tt_rank_reduce(tt_mat_mat_mul(V_tt, tt_transpose(V_tt)))
        Delta_tt = _tt_ipm_newton_step(obj_tt, linear_op_tt, bias_tt, XZ_tt, Y_tt)
        #Delta_tt = _symmetrisation(Delta_tt)
        #Delta_Y_tt = _tt_get_block(1, 0, Delta_tt)
        #Delta_XZ_tt = _get_xz_block(Delta_tt)
        #V_tt = _tt_homotopy_step(XZ_tt, Delta_XZ_tt)
        ## TODO: get step size that minimises dual residual
        #Y_tt = tt_rank_reduce(tt_add(Y_tt, Delta_Y_tt))
    return V_tt, Y_tt


if __name__ == "__main__":
    np.random.seed(1644)
    linear_op_tt = tt_rank_reduce(tt_random_gaussian([2], shape=(4, 2, 2)))
    initial_guess = tt_random_gaussian([2], shape=(2, 2))
    bias_tt = tt_mat(tt_linear_op(linear_op_tt, initial_guess), shape=(2, 2))
    obj_tt = tt_identity(len(bias_tt))
    _ = tt_ipm(obj_tt, linear_op_tt, bias_tt)

