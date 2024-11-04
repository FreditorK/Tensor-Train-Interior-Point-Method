import copy
import sys
import os

import numpy as np

sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from src.tt_ops import _tt_core_collapse, tt_random_gaussian
from src.tt_ipm import tt_ipm, _tt_get_block
import time


# Constraint 4 -----------------------------------------------------------------

def tt_partial_trace_op(block_size, dim):
    matrix_tt = [np.ones((1, 2, 2, 1)) for _ in range(dim-block_size)] + tt_identity(block_size)
    matrix_tt = tt_rank_reduce(tt_sub(matrix_tt, tt_identity(dim)))
    op_tt = tt_mask_to_linear_op(matrix_tt[:-block_size])
    for c in  matrix_tt[-block_size:]:
        core = np.zeros((c.shape[0], 4, *c.shape[1:]))
        core[:, 1] = c
        op_tt.append(core)
    return op_tt

def tt_partial_trace_op_adj(block_size, dim):
    matrix_tt = [np.ones((1, 2, 2, 1)) for _ in range(dim-block_size)] + [np.array([[0.0, 1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(block_size)]
    matrix_tt = tt_rank_reduce(tt_sub(matrix_tt, tt_identity(dim-block_size) + [np.array([[0.0, 1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(block_size)]))
    op_tt = tt_mask_to_linear_op(matrix_tt[:-block_size])
    for c in matrix_tt[-block_size:]:
        core = np.zeros((c.shape[0], 4, *c.shape[1:]))
        core[:, 0, 0, 1] = c[:, 0, 1]
        core[:, 3, 0, 1] = c[:, 0, 1]
        op_tt.append(core)
    return op_tt

# ------------------------------------------------------------------------------
# Constraint 5 -----------------------------------------------------------------

def tt_partial_J_trace_op(block_size, dim):
    op_tt = []
    core = np.zeros((1, 4, 2, 2, 2))
    core[0, 1, 0, 0, 0] = 1
    core[0, 2, 1, 0, 0] = 1
    core[0, 1, 0, 1, 1] = 1
    core[0, 2, 1, 1, 1] = 1
    op_tt.append(core)
    for _ in range(dim-block_size-1):
        core = np.zeros((2, 4, 2, 2, 2))
        core[:, 0, 0, 0] = np.eye(2)
        core[:, 1, 0, 1] = np.eye(2)
        core[:, 2, 1, 0] = np.eye(2)
        core[:, 3, 1, 1] = np.eye(2)
        op_tt.append(core)
    for _ in range(block_size-1):
        core = np.zeros((2, 4, 2, 2, 2))
        core[0, 0, :, :, 0] = 1
        core[1, 3, :, :, 1] = 1
        op_tt.append(core)
    core = np.zeros((2, 4, 2, 2, 1))
    core[0, 0] = 1
    core[1, 3] = 1
    op_tt.append(core)
    return op_tt

def tt_partial_J_trace_op_adj(block_size, dim):
    op_tt = []
    core = np.zeros((1, 4, 2, 2, 2))
    core[0, 0, 0, 1, 0] = 1
    core[0, 2, 1, 0, 0] = 1
    core[0, 1, 0, 1, 1] = 1
    core[0, 3, 1, 0, 1] = 1
    op_tt.append(core)
    for _ in range(dim - block_size - 1):
        core = np.zeros((2, 4, 2, 2, 2))
        core[:, 0, 0, 0] = np.eye(2)
        core[:, 1, 0, 1] = np.eye(2)
        core[:, 2, 1, 0] = np.eye(2)
        core[:, 3, 1, 1] = np.eye(2)
        op_tt.append(core)
    for _ in range(block_size - 1):
        core = np.zeros((2, 4, 2, 2, 2))
        core[0, :, 0, 0, 0] = 1
        core[1, :, 1, 1, 1] = 1
        op_tt.append(core)
    core = np.zeros((2, 4, 2, 2, 1))
    core[0, :, 0, 0] = 1
    core[1, :, 1, 1] = 1
    op_tt.append(core)
    return op_tt
# ------------------------------------------------------------------------------
# Constraint 6 -----------------------------------------------------------------

def tt_diag_block_sum_linear_op(block_size, dim):
    op_tt = []
    for _ in range(dim - block_size):
        core = np.zeros((1, 4, 2, 2, 1))
        core[0, 0, :, :, 0] = np.eye(2)
        op_tt.append(core)
    for _ in range(block_size):
        core = np.zeros((1, 4, 2, 2, 1))
        core[0, 0, 0, 0] = 1
        core[0, 1, 0, 1] = 1
        core[0, 2, 1, 0] = 1
        core[0, 3, 1, 1] = 1
        op_tt.append(core)
    return op_tt


def tt_diag_block_sum_linear_op_adj(block_size, dim):
    op_tt = []
    for _ in range(dim - block_size):
        core = np.zeros((1, 4, 2, 2, 1))
        core[0, 0, 0, 0] = 1
        core[0, 3, 0, 0] = 1
        op_tt.append(core)
    for _ in range(block_size):
        core = np.zeros((1, 4, 2, 2, 1))
        core[0, 0, 0, 0] = 1
        core[0, 1, 0, 1] = 1
        core[0, 2, 1, 0] = 1
        core[0, 3, 1, 1] = 1
        op_tt.append(core)
    return op_tt
# ------------------------------------------------------------------------------
# Constraint 7 -----------------------------------------------------------------

def tt_Q_m_P_op(dim):
    v_matrix_1 = [np.array([[1, 0], [0, 0]]).reshape(1, 2, 2, 1)] + tt_identity(dim)
    v_matrix_2 = [np.array([[0.0, -1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)] + [
        np.array([[1.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(dim)]
    matrix_tt = tt_add(v_matrix_1, v_matrix_2)
    op_tt = []
    core = np.zeros((matrix_tt[0].shape[0], 4, *matrix_tt[0].shape[1:]))
    core[:, 1] = matrix_tt[0]
    op_tt.append(core)
    for c in matrix_tt[1:]:
        core = np.zeros((c.shape[0], 4, *c.shape[1:]))
        core[:, 0, 0, 0] = c[:, 0, 0]
        core[:, 1, 0, 1] = c[:, 0, 1]
        core[:, 2, 1, 0] = c[:, 1, 0]
        core[:, 2, 1, 1] = c[:, 1, 1]
        op_tt.append(core)
    return tt_rank_reduce(op_tt)


def tt_Q_m_P_op_adj(dim):
    op_part_1_tt = []
    core = np.zeros((1, 4, 2, 2, 1))
    core[:, 1, 0, 1] = -0.5
    op_part_1_tt.append(core)
    for i in range(dim):
        core = np.zeros((1, 4, 2, 2, 1))
        core[:, 0, 0, 0] = 1
        core[:, 2, 1, 0] = 1
        op_part_1_tt.append(core)
    op_part_2_tt = []
    core = np.zeros((1, 4, 2, 2, 1))
    core[:, 2, 0, 1] = -0.5
    op_part_2_tt.append(core)
    for i in range(dim):
        core = np.zeros((1, 4, 2, 2, 1))
        core[:, 0, 0, 0] = 1
        core[:, 1, 1, 0] = 1
        op_part_2_tt.append(core)
    op_part_3_tt = []
    core = np.zeros((1, 4, 2, 2, 1))
    core[:, 0, 0, 1] = 1
    op_part_3_tt.append(core)
    for i in range(dim):
        core = np.zeros((1, 4, 2, 2, 1))
        core[:, 0, 0, 0] = 1
        core[:, 3, 1, 0] = 1
        op_part_3_tt.append(core)

    op_tt = tt_add(tt_add(op_part_1_tt, op_part_2_tt), op_part_3_tt)
    return tt_rank_reduce(op_tt)

# ------------------------------------------------------------------------------
# Constraint 8 -----------------------------------------------------------------


def tt_DS_op(block_size, dim):
    row_op = []
    core = np.zeros((1, 4, 2, 2, 1))
    core[:, 2, 1, 0] = 1
    row_op.append(core)
    for _ in range(dim-block_size):
        core = np.zeros((1, 4, 2, 2, 1))
        core[:, 0, 0, 0] = 1
        core[:, 1, 0, 1] = 1
        row_op.append(core)
    for _ in range(block_size):
        core = np.zeros((1, 4, 2, 2, 1))
        core[:, 0, 0, 0] = 1
        core[:, 0, 0, 1] = 1
        row_op.append(core)
    col_op = []
    core = np.zeros((1, 4, 2, 2, 1))
    core[:, 0, 1, 0] = 1
    col_op.append(core)
    for _ in range(dim - block_size):
        core = np.zeros((1, 4, 2, 2, 1))
        core[:, 3, 0, 0] = 1
        core[:, 3, 0, 1] = 1
        col_op.append(core)
    for _ in range(block_size):
        core = np.zeros((1, 4, 2, 2, 1))
        core[:, 0, 0, 0] = 1
        core[:, 3, 0, 1] = 1
        col_op.append(core)
    op_tt = tt_rank_reduce(tt_add(row_op, col_op))
    return op_tt


def tt_DS_op_adj(block_size, dim):
    row_op_1 = []
    core = np.zeros((1, 4, 2, 2, 1))
    core[:, 2, 1, 0] = 1
    row_op_1.append(core)
    for _ in range(dim - block_size):
        core = np.zeros((1, 4, 2, 2, 1))
        core[:, 0, 0, 0] = 1
        core[:, 1, 0, 1] = 1
        row_op_1.append(core)
    for _ in range(block_size):
        core = np.zeros((1, 4, 2, 2, 1))
        core[:, 0, 0, 0] = 1
        core[:, 1, 0, 0] = 1
        row_op_1.append(core)

    row_op_2 = []
    core = np.zeros((1, 4, 2, 2, 1))
    core[:, 1, 1, 0] = 1
    row_op_2.append(core)
    for _ in range(dim - block_size):
        core = np.zeros((1, 4, 2, 2, 1))
        core[:, 0, 0, 0] = 1
        core[:, 2, 0, 1] = 1
        row_op_2.append(core)
    for _ in range(block_size):
        core = np.zeros((1, 4, 2, 2, 1))
        core[:, 0, 0, 0] = 1
        core[:, 2, 0, 0] = 1
        row_op_2.append(core)
    row_op = tt_add(row_op_1, row_op_2)

    col_op_1 = []
    core = np.zeros((1, 4, 2, 2, 1))
    core[:, 1, 1, 0] = 1
    col_op_1.append(core)
    for _ in range(dim - block_size):
        core = np.zeros((1, 4, 2, 2, 1))
        core[:, 0, 0, 0] = 1
        core[:, 0, 0, 1] = 1
        core[:, 2, 0, 0] = 1
        core[:, 2, 0, 1] = 1
        col_op_1.append(core)
    for _ in range(block_size):
        core = np.zeros((1, 4, 2, 2, 1))
        core[:, 0, 0, 0] = 1
        core[:, 2, 0, 1] = 1
        col_op_1.append(core)

    col_op_2 = []
    core = np.zeros((1, 4, 2, 2, 1))
    core[:, 2, 1, 0] = 1
    col_op_2.append(core)
    for _ in range(dim - block_size):
        core = np.zeros((1, 4, 2, 2, 1))
        core[:, 0, 0, 0] = 1
        core[:, 0, 0, 1] = 1
        core[:, 1, 0, 0] = 1
        core[:, 1, 0, 1] = 1
        col_op_2.append(core)
    for _ in range(block_size):
        core = np.zeros((1, 4, 2, 2, 1))
        core[:, 0, 0, 0] = 1
        core[:, 1, 0, 1] = 1
        col_op_2.append(core)
    col_op = tt_add(col_op_1, col_op_2)

    op_tt = tt_rank_reduce(tt_add(row_op, col_op))
    return op_tt

def tt_DS_bias(block_size, dim):
    matrix_tt_1 = (
            [np.array([[0.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1)]
            + [np.array([[1.0, 1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(dim - block_size)]
            + [np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(block_size)]
    )
    matrix_tt_2 = (
            [np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)]
            + [np.array([[0.0, 0.0], [0.0, 1.0]]).reshape(1, 2, 2, 1) for _ in range(dim - block_size)]
            + [np.eye(2).reshape(1, 2, 2, 1) for _ in range(block_size)]
    )
    return tt_rank_reduce(tt_add(matrix_tt_1, matrix_tt_2))

# ------------------------------------------------------------------------------
# Constraint 9 -----------------------------------------------------------------

def tt_padding_op(dim):
    matrix_tt_1 = [np.array([[0.0, 1.0], [1.0, 1.0]]).reshape(1, 2, 2, 1)] + [np.ones((1, 2, 2, 1)) for _ in range(dim-1)]
    matrix_tt_2 = [np.array([[0.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1)] +  [np.array([[1.0, 1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1) for _ in
                                                                              range(dim - 1)]
    matrix_tt_3 = [np.array([[0.0, 1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)] + [
        np.array([[1.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1) for _ in
        range(dim - 1)]
    matrix_tt = tt_rank_reduce(tt_sub(tt_sub(matrix_tt_1, matrix_tt_2), matrix_tt_3))
    return tt_mask_to_linear_op(matrix_tt)

def tt_padding_op_adjoint(dim):
    matrix_tt_1 = [np.array([[0.0, 1.0], [1.0, 1.0]]).reshape(1, 2, 2, 1)] + [np.ones((1, 2, 2, 1)) for _ in range(dim-1)]
    matrix_tt_2 = [np.array([[0.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1)] +  [np.array([[1.0, 1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1) for _ in
                                                                              range(dim - 1)]
    matrix_tt_3 = [np.array([[0.0, 1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)] + [
        np.array([[1.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1) for _ in
        range(dim - 1)]
    matrix_tt = tt_rank_reduce(tt_sub(tt_sub(matrix_tt_1, matrix_tt_2), matrix_tt_3))
    return tt_mask_to_linear_op_adjoint(matrix_tt)

# ------------------------------------------------------------------------------

@dataclass
class Config:
    seed = 4
    max_rank = 3

if __name__ == "__main__":
    print("Creating Problem...")
    """
        [Q   P  0 ]
    X = [P^T 1  0 ]
        [0   0  I ]
    e.g.
        [Q_11 Q_12 Q_13 Q_14 | P_11 |   0    0    0]
        [Q_21 Q_22 Q_23 Q_24 | P_21 |   0    0    0]
        [Q_31 Q_32 Q_33 Q_34 | P_12 |   0    0    0]
        [Q_41 Q_42 Q_43 Q_44 | P_22 |   0    0    0]
    X = [------------------------------------------]
        [P_11 P_21 P_12 P_22 |    1 |   0    0    0]
        [------------------------------------------]
        [   0    0    0    0 |    0 |   1    0    0]
        [   0    0    0    0 |    0 |   0    1    0]
        [   0    0    0    0 |    0 |   0    0    1]
        
        
        [ 6  6  | 5  4  | 7 | P P P]
        [ 6  6  | 0  5  | 7 | P P P]
        [--------------------------]
        [ 5  4  | 8c  0 | 7 | P P P]
        [ 0  5  | 0  8c | 7 | P P P]
    Y = [--------------------------]
        [ 8r 0  | 8r 0  | P | P P P]
        [--------------------------]
        [ P  P  | P  P  | P | P P P]
        [ P  P  | P  P  | P | P P P]
        [ P  P  | P  P  | P | P P P]
    """
    np.set_printoptions(linewidth=600, threshold=np.inf, precision=4, suppress=True)
    print("Creating Problem...")
    q_op_prefix = np.zeros((1, 4, 2, 2, 1))
    q_op_prefix[0, 0, 0, 0, 0] = 1
    padding_op_prefix = np.zeros((1, 4, 2, 2, 1))
    padding_op_prefix[0, 3, 1, 1, 0] = 1
    q_bias_prefix = np.zeros((1, 2, 2, 1))
    q_bias_prefix[0, 0, 0, 0] = 1
    padding_bias_prefix = np.zeros((1, 2, 2, 1))
    padding_bias_prefix[0, 1, 1, 0] = 1

    n = 2

    np.random.seed(Config.seed)
    G_A = tt_random_graph(n, Config.max_rank)
    print(np.round(tt_matrix_to_matrix(G_A), decimals=2))
    print(f"TT-Ranks: {tt_ranks(G_A)}")

    G_B = tt_random_graph(n, Config.max_rank)
    print(np.round(tt_matrix_to_matrix(G_B), decimals=2))
    print(f"TT-Ranks: {tt_ranks(G_B)}")

    C_tt = tt_rank_reduce([q_bias_prefix] + tt_kron(G_B, G_A))
    n_sq = len(C_tt) - 1


    # Equality Operator
    # IV
    partial_tr_op = [q_op_prefix] + tt_partial_trace_op(n, n_sq)
    partial_tr_op_adj = [q_op_prefix] + tt_partial_trace_op_adj(n, n_sq)
    partial_tr_op_bias = tt_zero_matrix(n_sq+1)

    L_op_tt = partial_tr_op
    L_op_tt_adj = partial_tr_op_adj
    eq_bias_tt = partial_tr_op_bias
    # ---
    # V
    partial_tr_J_op = [q_op_prefix] + tt_partial_J_trace_op(n, n_sq)
    partial_tr_J_op_adj = [q_op_prefix] + tt_partial_J_trace_op_adj(n, n_sq)
    partial_tr_J_op_bias = [q_bias_prefix] + tt_add(
        [np.array([[0.0, 1.0], [1.0, 0.0]]).reshape(1, 2, 2, 1)] + tt_one_matrix(n_sq-n-1) + [np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(n)],
        [np.array([[0.0, 1.0], [1.0, 0.0]]).reshape(1, 2, 2, 1)] + tt_one_matrix(n_sq - n - 1) + [np.array([[0.0, 0.0], [0.0, 1.0]]).reshape(1, 2, 2, 1) for _ in range(n)]
    )

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, partial_tr_J_op))
    L_op_tt_adj = tt_rank_reduce(tt_add(L_op_tt_adj, partial_tr_J_op_adj))
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, partial_tr_J_op_bias))
    # ---
    # VI
    diag_block_sum_op = [q_op_prefix] + tt_diag_block_sum_linear_op(n, n_sq)
    diag_block_sum_op_adj = [q_op_prefix] + tt_diag_block_sum_linear_op_adj(n, n_sq)
    diag_block_sum_op_bias = [q_bias_prefix] + [np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(n_sq-n)] + tt_identity(n)

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, diag_block_sum_op))
    L_op_tt_adj = tt_rank_reduce(tt_add(L_op_tt_adj, diag_block_sum_op_adj))
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, diag_block_sum_op_bias))
    # ---
    # VII
    # FIXME: AMeN error moderately high
    Q_m_P_op = tt_Q_m_P_op(n_sq)
    Q_m_P_op_adj = tt_Q_m_P_op_adj(n_sq)
    Q_m_P_op_bias = tt_zero_matrix(n_sq+1)

    #L_op_tt = tt_rank_reduce(tt_add(L_op_tt, Q_m_P_op))
    #L_op_tt_adj = tt_rank_reduce(tt_add(L_op_tt_adj, Q_m_P_op_adj))
    #eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, Q_m_P_op_bias))
    # ---
    # VIII
    # FIXME: AMeN error very high
    DS_op = tt_DS_op(n, n_sq)
    DS_op_adj = tt_DS_op_adj(n, n_sq)
    DS_op_bias = tt_DS_bias(n, n_sq)

    #L_op_tt = tt_rank_reduce(tt_add(L_op_tt, DS_op))
    #L_op_tt_adj = tt_rank_reduce(tt_add(L_op_tt_adj, DS_op_adj))
    #eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, DS_op_bias))
    # ---
    # IX
    padding_op = tt_padding_op(n_sq+1)
    padding_op_adj = tt_padding_op_adjoint(n_sq + 1)
    padding_op_bias = [padding_bias_prefix] + tt_identity(n_sq)

    #L_op_tt = tt_rank_reduce(tt_add(L_op_tt, padding_op))
    #L_op_tt_adj = tt_rank_reduce(tt_add(L_op_tt_adj, padding_op_adj))
    #eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, padding_op_bias))
    # ---
    # Inequality Operator
    # X
    Q_ineq_op = [q_op_prefix] + tt_mask_to_linear_op(tt_one_matrix(n_sq))
    Q_ineq_op_adj = [q_op_prefix] + tt_mask_to_linear_op_adjoint(tt_one_matrix(n_sq))
    Q_ineq_bias = tt_zero_matrix(n_sq+1)
    # ---
    print("...Problem created!")
    print(f"Objective TT-ranks: {tt_ranks(C_tt)}")
    print(f"Eq Op-rank: {tt_ranks(L_op_tt)}")
    print(f"Eq Op-adjoint-rank: {tt_ranks(L_op_tt)}")
    print(f"Eq Bias-rank: {tt_ranks(eq_bias_tt)}")
    print("-----------------------------------")
    print(f"Ineq Op-rank: {tt_ranks(Q_ineq_op)}")
    print(f"Ineq Op-adjoint-rank: {tt_ranks(Q_ineq_op_adj)}")
    print(f"Ineq Bias-rank: {tt_ranks(Q_ineq_bias)}")
    #print(np.round(tt_matrix_to_matrix(tt_mat(tt_linear_op(L_op_tt, tt_random_gaussian([3, 3, 3, 3], shape=(2, 2))), shape=(2, 2))), decimals=2))
    #print(np.round(tt_matrix_to_matrix(eq_bias_tt), decimals=2))
    t0 = time.time()
    X_tt, Y_tt, T_tt, Z_tt = tt_ipm(
        C_tt,
        L_op_tt,
        L_op_tt_adj,
        eq_bias_tt,
        max_iter=2,
        verbose=True
    )
    t1 = time.time()
    print("Solution: ")
    #print(np.round(tt_matrix_to_matrix(X_tt), decimals=2))
    print(f"Objective value: {tt_inner_prod(C_tt, X_tt)}")
    print("Complementary Slackness: ", tt_inner_prod(X_tt, Z_tt))
    print(f"Ranks X_tt: {tt_ranks(X_tt)}, Y_tt: {tt_ranks(Y_tt)}, \n "
          f"     T_tt: {tt_ranks(T_tt)}, Z_tt: {tt_ranks(Z_tt)} ")
    print(f"Time: {t1 - t0}s")
