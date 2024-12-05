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


Q_PREFIX = [np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1), np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)]


# Constraint 4 -----------------------------------------------------------------

def tt_partial_trace_op(block_size, dim):
    matrix_tt = [np.ones((1, 2, 2, 1)) for _ in range(dim-block_size)] + tt_identity(block_size)
    matrix_tt = tt_sub(matrix_tt, tt_identity(dim))
    basis = []
    for c in tt_vec(matrix_tt[:-block_size]):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, 0, 0] = c[:, 0]
        core[:, 1, 1] = c[:, 1]
        basis.append(core)
    for c in matrix_tt[-block_size:]:
        core = np.zeros((c.shape[0], 2, 2, 2, 2, c.shape[-1]))
        core[:, 0, :, 1, :] = c # (0, 1) <-
        basis.extend(break_core_bond(core))
    return Q_PREFIX + tt_rank_reduce(basis)

def tt_partial_trace_op_adj(block_size, dim):
    matrix_tt = [np.ones((1, 2, 2, 1)) for _ in range(dim-block_size)]
    matrix_tt = tt_sub(matrix_tt, tt_identity(dim-block_size))
    basis = []
    for c in tt_vec(matrix_tt):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, 0, 0] = c[:, 0]
        core[:, 1, 1] = c[:, 1]
        basis.append(core)
    for _ in  range(block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 0, 0, 1] = 1 # (0, 0) <- (0, 1)
        core[:, 1, 0, 1, 1] = 1 # (1, 1) <- (0, 1)
        basis.extend(break_core_bond(core))
    return Q_PREFIX + tt_rank_reduce(basis)

# ------------------------------------------------------------------------------
# Constraint 5 -----------------------------------------------------------------

def tt_partial_J_trace_op(block_size, dim):
    op_tt = []
    core = np.zeros((1, 2, 2, 2, 2, 2))
    core[0, 0, 0, 1, 0, 0] = 1 # (0, 1) <- (0, 0)
    core[0, 1, 1, 0, 0, 0] = 1 # (1, 0) <- (1, 0)
    core[0, 0, 0, 1, 1, 1] = 1 # (0, 1) <- (0, 1)
    core[0, 1, 1, 0, 1, 1] = 1 # (1, 0) <- (1, 1)
    op_tt.extend(break_core_bond(core))
    for _ in range(dim-block_size-1):
        core = np.zeros((2, 2, 2, 2, 2, 2))
        core[:, 0, 0, 0, 0] = np.eye(2) # (0, 0) <- (0, 0)
        core[:, 0, 0, 1, 1] = np.eye(2) # (0, 1) <- (0, 1)
        core[:, 1, 1, 0, 0] = np.eye(2) # (1, 0) <- (1, 0)
        core[:, 1, 1, 1, 1] = np.eye(2) # (1, 1) <- (1, 1)
        op_tt.extend(break_core_bond(core))
    for _ in range(block_size-1):
        core = np.zeros((2, 2, 2, 2, 2, 2))
        core[0, 0, :, 0, :, 0] = 1
        core[1, 1, :, 1, :, 1] = 1
        op_tt.extend(break_core_bond(core))
    core = np.zeros((2, 2, 2, 2, 2, 1))
    core[0, 0, :, 0] = 1
    core[1, 1, :, 1] = 1
    op_tt.extend(break_core_bond(core))
    return Q_PREFIX + tt_rank_reduce(op_tt)

def tt_partial_J_trace_op_adj(block_size, dim):
    op_tt = []
    core = np.zeros((1, 2, 2, 2, 2, 2))
    core[0, 0, 0, 0, 1, 0] = 1 # (0, 0) <- (0, 1)
    core[0, 1, 1, 0, 0, 0] = 1 # (1, 0) <- (1, 0)
    core[0, 0, 0, 1, 1, 1] = 1 # (0, 1) <- (0, 1)
    core[0, 1, 1, 1, 0, 1] = 1 # (1, 1) <- (1, 0)
    op_tt.extend(break_core_bond(core))
    for _ in range(dim - block_size - 1):
        core = np.zeros((2, 2, 2, 2, 2, 2))
        core[:, 0, 0, 0, 0] = np.eye(2) # (0, 0) <- (0, 0)
        core[:, 0, 0, 1, 1] = np.eye(2) # (0, 1) <- (0, 1)
        core[:, 1, 1, 0, 0] = np.eye(2) # (1, 0) <- (1, 0)
        core[:, 1, 1, 1, 1] = np.eye(2) # (1, 1) <- (1, 1)
        op_tt.extend(break_core_bond(core))
    for _ in range(block_size - 1):
        core = np.zeros((2, 2, 2, 2, 2, 2))
        core[0, :, 0, :, 0, 0] = 1  # (:, 0) <- (0, 0)
        core[1, :, 1, :, 1, 1] = 1 # (:, 0) <- (1, 1)
        op_tt.extend(break_core_bond(core))
    core = np.zeros((2, 2, 2, 2, 2, 1))
    core[0, :, 0, :, 0] = 1 # (:, 0) <- (0, 0)
    core[1, :, 1, :, 1] = 1 # (:, 0) <- (1, 1)

    op_tt.extend(break_core_bond(core))
    return Q_PREFIX + tt_rank_reduce(op_tt)
# ------------------------------------------------------------------------------
# Constraint 6 -----------------------------------------------------------------

def tt_diag_block_sum_linear_op(block_size, dim):
    op_tt = []
    for _ in range(dim - block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[0, 0, :, 0, :, 0] = np.eye(2)
        op_tt.extend(break_core_bond(core))
    for _ in range(block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 0, 0, 0] = 1 # (0, 0) <- (0, 0)
        core[:, 0, 0, 1, 1] = 1 # (0, 1) <- (0, 1)
        core[:, 1, 1, 0, 0] = 1 # (1, 0) <- (1, 0)
        core[:, 1, 1, 1, 1] = 1 # (1, 1) <- (1, 1)
        op_tt.extend(break_core_bond(core))
    return Q_PREFIX + tt_rank_reduce(op_tt)


def tt_diag_block_sum_linear_op_adj(block_size, dim):
    op_tt = []
    for _ in range(dim - block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 0, 0, 0] = 1 # (0, 0) <- (0, 0)
        core[:, 1, 0, 1, 0] = 1 # (1, 1) <- (0, 0)
        op_tt.extend(break_core_bond(core))
    for _ in range(block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 0, 0, 0] = 1 # (0, 0) <- (0, 0)
        core[:, 0, 0, 1, 1] = 1 # (0, 1) <- (0, 1)
        core[:, 1, 1, 0, 0] = 1 # (1, 0) <- (1, 0)
        core[:, 1, 1, 1, 1] = 1 # (1, 1) <- (1, 1)
        op_tt.extend(break_core_bond(core))
    return Q_PREFIX + tt_rank_reduce(op_tt)
# ------------------------------------------------------------------------------
# Constraint 7 -----------------------------------------------------------------

def tt_Q_m_P_op(dim):
    v_matrix_1 = [np.array([[1, 0], [0, 0]]).reshape(1, 2, 2, 1)] + tt_identity(dim)
    v_matrix_2 = [np.array([[0.0, -1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)] + [
        np.array([[1.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(dim)]
    matrix_tt = tt_add(v_matrix_1, v_matrix_2)
    op_tt = []
    core = np.zeros((1, 2, 2, 2, 2, matrix_tt[0].shape[-1]))
    core[:, 0, :, 1] = matrix_tt[0]
    op_tt.extend(break_core_bond(core))
    for c in matrix_tt[1:]:
        core = np.zeros((c.shape[0], 2, 2, 2, 2, c.shape[-1]))
        core[:, 0, 0, 0, 0] = c[:, 0, 0] # (0, 0) <- (0, 0)
        core[:, 0, 0, 1, 1] = c[:, 0, 1] # (0, 1) <- (0, 1)
        core[:, 1, 1, 0, 0] = c[:, 1, 0] # (1, 0) <- (1, 0)
        core[:, 1, 1, 0, 1] = c[:, 1, 1] # (1, 0) <- (1, 1)
        op_tt.extend(break_core_bond(core))
    return tt_rank_reduce(op_tt)


def tt_Q_m_P_op_adj(dim):
    op_part_1_tt = []
    core = np.zeros((1, 2, 2, 2, 2, 1))
    core[:, 0, 0, 1, 1] = -0.5 # (0, 1) <- (0, 1)
    op_part_1_tt.extend(break_core_bond(core))
    for i in range(dim):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 0, 0, 0] = 1 # (0, 0) <- (0, 0)
        core[:, 1, 1, 0, 0] = 1 # (1, 0) <- (1, 0)
        op_part_1_tt.extend(break_core_bond(core))
    op_part_2_tt = []
    core = np.zeros((1, 2, 2, 2, 2, 1))
    core[:, 1, 0, 0, 1] = -0.5 # (1, 0) <- (0, 1)
    op_part_2_tt.extend(break_core_bond(core))
    for i in range(dim):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 0, 0, 0] = 1 # (0, 0) <- (0, 0)
        core[:, 0, 1, 1, 0] = 1 # (0, 1) <- (1, 0)
        op_part_2_tt.extend(break_core_bond(core))
    op_part_3_tt = []
    core = np.zeros((1, 2, 2, 2, 2, 1))
    core[:, 0, 0, 0, 1] = 1 # (0, 0) <- (0, 1)
    op_part_3_tt.extend(break_core_bond(core))
    for i in range(dim):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 0, 0, 0] = 1 # (0, 0) <- (0, 0)
        core[:, 1, 1, 1, 0] = 1 # (1, 1) <- (1, 0)
        op_part_3_tt.extend(break_core_bond(core))

    op_tt = tt_add(tt_add(op_part_1_tt, op_part_2_tt), op_part_3_tt)
    return tt_rank_reduce(op_tt)

# ------------------------------------------------------------------------------
# Constraint 8 -----------------------------------------------------------------


def tt_DS_op(block_size, dim):
    row_op = []
    core = np.zeros((1, 2, 2, 2, 2, 1))
    core[:, 1, 1, 0, 0] = 1 # (1, 0) <- (1, 0)
    row_op.extend(break_core_bond(core))
    for _ in range(dim-block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 0, 0, 0] = 1 # (0, 0) <- (0, 0)
        core[:, 0, 0, 1, 1] = 1 # (0, 1) <- (0, 1)
        row_op.extend(break_core_bond(core))
    for _ in range(block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 0, 0, 0] = 1 # (0, 0) <- (0, 0)
        core[:, 0, 0, 0, 1] = 1 # (0, 0) <- (0, 1)
        row_op.extend(break_core_bond(core))
    col_op = []
    core = np.zeros((1, 2, 2, 2, 2, 1))
    core[:, 0, 1, 0, 0] = 1 # (0, 0) <- (1, 0)
    col_op.extend(break_core_bond(core))
    for _ in range(dim - block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 1, 0, 1, 0] = 1 # (1, 1) <- (0, 0)
        core[:, 1, 0, 1, 1] = 1 # (1, 1) <- (0, 1)
        col_op.extend(break_core_bond(core))
    for _ in range(block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 0, 0, 0] = 1 # (0, 0) <- (0, 0)
        core[:, 1, 0, 1, 1] = 1 # (1, 1) <- (0, 1)
        col_op.extend(break_core_bond(core))
    op_tt = tt_rank_reduce(tt_add(row_op, col_op))
    return op_tt


def tt_DS_op_adj(block_size, dim):
    row_op_1 = []
    core = np.zeros((1, 2, 2, 2, 2, 1))
    core[:, 1, 1, 0, 0] = 1 # (1, 0) <- (1, 0)
    row_op_1.extend(break_core_bond(core))
    for _ in range(dim - block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 0, 0, 0] = 1 # (0, 0) <- (0, 0)
        core[:, 0, 0, 1, 1] = 1 # (0, 1) <- (0, 1)
        row_op_1.extend(break_core_bond(core))
    for _ in range(block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 0, 0, 0] = 1 # (0, 0) <- (0, 0)
        core[:, 0, 0, 1, 0] = 1 # (0, 1) <- (0, 0)
        row_op_1.extend(break_core_bond(core))

    row_op_2 = []
    core = np.zeros((1, 2, 2, 2, 2, 1))
    core[:, 0, 1, 1, 0] = 1 # (0, 1) <- (1, 0)
    row_op_2.extend(break_core_bond(core))
    for _ in range(dim - block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 0, 0, 0] = 1 # (0, 0) <- (0, 0)
        core[:, 1, 0, 0, 1] = 1 # (1, 0) <- (0, 1)
        row_op_2.extend(break_core_bond(core))
    for _ in range(block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 0, 0, 0] = 1 # (0, 0) <- (0, 0)
        core[:, 1, 0, 0, 0] = 1 # (1, 0) <- (0, 0)
        row_op_2.extend(break_core_bond(core))
    row_op = tt_add(row_op_1, row_op_2)

    col_op_1 = []
    core = np.zeros((1, 2, 2, 2, 2, 1))
    core[:, 0, 0, 1, 0] = 1 # (0, 1) <- (0, 0)
    col_op_1.extend(break_core_bond(core))
    for _ in range(dim - block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 1, 0, 1] = 1 # (0, 0) <- (1, 1)
        core[:, 1, 1, 0, 1] = 1 # (1, 0) <- (1, 1)
        col_op_1.extend(break_core_bond(core))
    for _ in range(block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 0, 0, 0] = 1 # (0, 0) <- (0, 0)
        core[:, 1, 1, 0, 1] = 1 # (1, 0) <- (1, 1)
        col_op_1.extend(break_core_bond(core))

    col_op_2 = []
    core = np.zeros((1, 2, 2, 2, 2, 1))
    core[:, 1, 0, 0, 0] = 1 # (1, 0) <- (0, 0)
    col_op_2.extend(break_core_bond(core))
    for _ in range(dim - block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 1, 0, 1] = 1  # (0, 0) <- (1, 1)
        core[:, 0, 1, 1, 1] = 1  # (0, 1) <- (1, 1)
        col_op_2.extend(break_core_bond(core))
    for _ in range(block_size):
        core = np.zeros((1, 2, 2, 2, 2, 1))
        core[:, 0, 0, 0, 0] = 1 # (0, 0) <- (0, 0)
        core[:, 0, 1, 1, 1] = 1 # (0, 1) <- (1, 1)
        col_op_2.extend(break_core_bond(core))
    col_op = tt_add(col_op_1, col_op_2)

    op_tt = tt_rank_reduce(tt_add(row_op, col_op))
    # FIXME: row_op is col_op and col_op is row_op, doesn't change anything but out of principle
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
    matrix_tt = [np.array([[0.0, 0.0], [0.0, 1.0]]).reshape(1, 2, 2, 1)] + [np.eye(2).reshape(1, 2, 2, 1) for _ in range(dim)]
    basis = []
    for c in tt_vec(matrix_tt):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, 0, 0] = c[:, 0]
        core[:, 1, 1] = c[:, 1]
        basis.append(core)
    return tt_rank_reduce(basis)

def tt_padding_op_adj(dim):
    return tt_padding_op(dim)

# ------------------------------------------------------------------------------
# Constraint 10 ----------------------------------------------------------------
def tt_ineq_op(dim):
    matrix_tt = [-np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)] + [np.array([[1.0, 1.0], [1.0, 1.0]]).reshape(1, 2, 2, 1) for _ in range(dim)]
    matrix_tt = tt_add(matrix_tt, [-np.array([[0.0, 1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)] + [np.array([[1.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(dim)])
    matrix_tt = tt_add(matrix_tt, [-np.array([[0.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1)] + [np.array([[1.0, 1.0], [0.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(dim)])
    basis = tt_matrix_to_mask_op(matrix_tt)
    return tt_rank_reduce(basis)

def tt_ineq_op_adj(dim):
    return tt_ineq_op(dim)
# ------------------------------------------------------------------------------

@dataclass
class Config:
    seed = 5
    max_rank = 3

if __name__ == "__main__":
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
        [ 5  4  | 8r  0 | 7 | P P P]
        [ 0  5  | 0  8r | 7 | P P P]
    Y = [--------------------------]
        [ 8c 0  | 8c 0  | P | P P P]
        [--------------------------]
        [ P  P  | P  P  | P | P P P]
        [ P  P  | P  P  | P | P P P]
        [ P  P  | P  P  | P | P P P]
    """
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    print("Creating Problem...")

    n = 1

    np.random.seed(Config.seed)
    G_A = tt_random_graph(n, Config.max_rank)
    print("Graph A: ")
    print(np.round(tt_matrix_to_matrix(G_A), decimals=2))

    G_B = tt_random_graph(n, Config.max_rank)
    print("Graph B: ")
    print(np.round(tt_matrix_to_matrix(G_B), decimals=2))

    print("Objective matrix: ")
    C_tt = [np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)] + tt_kron(G_B, G_A)
    print(np.round(tt_matrix_to_matrix(C_tt), decimals=2))

    # Equality Operator
    # IV
    partial_tr_op = tt_partial_trace_op(n, 2*n)
    partial_tr_op_adj = tt_partial_trace_op_adj(n, 2*n)
    partial_tr_op_bias = tt_zero_matrix(2 * n + 1)

    def test_partial_tr_op():
        random_A = tt_random_gaussian([3] * (2 * n), shape=(2, 2))
        M = tt_matrix_to_matrix(random_A)
        print(np.round(M, decimals=4))
        print(np.round(tt_matrix_to_matrix(tt_mat(tt_matrix_vec_mul(partial_tr_op, tt_vec(random_A)))), decimals=4))
        partial_traces = {}
        m = 2**n
        for i, j in product(range(m), range(m)):
            if i != j:
                partial_traces[(i, j)] = (np.trace(M[m*i:m*(i+1), m*j: m*(j+1)]))
        print(partial_traces)

    def test_partial_tr_op_adj():
        random_A = tt_random_gaussian([3] * (2 * n), shape=(2, 2))
        M = tt_matrix_to_matrix(random_A)
        print(np.round(M, decimals=4))
        print(np.round(tt_matrix_to_matrix(tt_mat(tt_matrix_vec_mul(partial_tr_op_adj, tt_vec(random_A)))), decimals=4))
        partial_traces = {}
        m = 2**n
        for i, j in product(range(m), range(m)):
            if i != j:
                partial_traces[(i, j)] = (M[m*i:m*(i+1), m*j: m*(j+1)][0, -1])
        print(partial_traces)

    L_op_tt = partial_tr_op
    L_op_tt_adj = partial_tr_op_adj
    eq_bias_tt = partial_tr_op_bias
    # ---
    # V
    partial_tr_J_op = tt_partial_J_trace_op(n, 2*n)
    partial_tr_J_op_adj = tt_partial_J_trace_op_adj(n, 2*n)
    partial_tr_J_op_bias = tt_add(
        [np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)] + [np.array([[0.0, 1.0], [1.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(n)] + [np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(n)],
        [np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)] + [np.array([[0.0, 1.0], [1.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(n)] + [np.array([[0.0, 0.0], [0.0, 1.0]]).reshape(1, 2, 2, 1) for _ in range(n)]
    )

    def test_partial_tr_J_op():
        random_A = tt_random_gaussian([3] * (2 * n), shape=(2, 2))
        M = tt_matrix_to_matrix(random_A)
        print(np.round(M, decimals=4))
        print(np.round(tt_matrix_to_matrix(tt_mat(tt_matrix_vec_mul(partial_tr_J_op, tt_vec(random_A)))), decimals=4))
        partial_traces = {}
        m = 2**n
        for i, j in product(range(m), range(m)):
            partial_traces[(i, j)] = (np.trace(np.ones_like(M[m*i:m*(i+1), m*j: m*(j+1)]) @ M[m*i:m*(i+1), m*j: m*(j+1)]))
        print(partial_traces)

    def test_partial_tr_J_op_adj():
        random_A = tt_random_gaussian([3] * (2 * n), shape=(2, 2))
        M = tt_matrix_to_matrix(random_A)
        print(np.round(M, decimals=4))
        print(np.round(tt_matrix_to_matrix(tt_mat(tt_matrix_vec_mul(partial_tr_J_op_adj, tt_vec(random_A)))), decimals=4))
        partial_traces = {}
        m = 2**n
        for i, j in product(range(m), range(m)):
            if i > 0 and j == 0 or i == 0 and j > 0:
                partial_traces[(i, j)] = [M[m*i:m*(i+1), m*j: m*(j+1)][0, 0], M[m*i:m*(i+1), m*j: m*(j+1)][-1, -1]]
        print(partial_traces)


    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, partial_tr_J_op))
    L_op_tt_adj = tt_rank_reduce(tt_add(L_op_tt_adj, partial_tr_J_op_adj))
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, partial_tr_J_op_bias))

    # ---
    # VI
    diag_block_sum_op = tt_diag_block_sum_linear_op(n, 2*n)
    diag_block_sum_op_adj = tt_diag_block_sum_linear_op_adj(n, 2*n)
    diag_block_sum_op_bias = [np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(n+1)] + tt_identity(n)

    def test_diag_block_sum_op():
        random_A = tt_random_gaussian([3] * (2 * n), shape=(2, 2))
        M = tt_matrix_to_matrix(random_A)
        print(np.round(M, decimals=4))
        print(np.round(tt_matrix_to_matrix(tt_mat(tt_matrix_vec_mul(diag_block_sum_op, tt_vec(random_A)))), decimals=4))
        partial_traces = 0
        m = 2**n
        for i in range(m):
            partial_traces += M[m*i:m*(i+1), m*i:m*(i+1)]
        print(partial_traces)

    def test_diag_block_sum_op_adj():
        random_A = tt_random_gaussian([3] * (2 * n), shape=(2, 2))
        M = tt_matrix_to_matrix(random_A)
        print(np.round(M, decimals=4))
        print(np.round(tt_matrix_to_matrix(tt_mat(tt_matrix_vec_mul(diag_block_sum_op_adj, tt_vec(random_A)))), decimals=4))
        m = 2**n
        print(M[0:m, 0:m])

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, diag_block_sum_op))
    L_op_tt_adj = tt_rank_reduce(tt_add(L_op_tt_adj, diag_block_sum_op_adj))
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, diag_block_sum_op_bias))

    # ---
    # VII
    Q_m_P_op = tt_Q_m_P_op(2*n)
    Q_m_P_op_adj = tt_Q_m_P_op_adj(2*n)
    Q_m_P_op_bias = tt_zero_matrix(2*n + 1)

    def test_Q_m_P_op():
        random_A = tt_random_gaussian([3] * (2 * n), shape=(2, 2))
        M = tt_matrix_to_matrix(random_A)
        print(np.round(M, decimals=4))
        print(np.round(tt_matrix_to_matrix(tt_mat(tt_matrix_vec_mul(Q_m_P_op, tt_vec(random_A)))), decimals=4))
        m = 2**(2*n)
        Q = M[:m, :m]
        P = M[:m, m]
        partial_traces = np.diagonal(Q) - P
        print(partial_traces)

    def test_Q_m_P_op_adj():
        random_A = tt_random_gaussian([3] * (2 * n), shape=(2, 2))
        M = tt_matrix_to_matrix(random_A)
        print(np.round(M, decimals=4))
        print(np.round(tt_matrix_to_matrix(tt_mat(tt_matrix_vec_mul(Q_m_P_op_adj, tt_vec(random_A)))), decimals=4))
        m = 2**(2*n)
        P = M[:m, m]
        partial_traces = P
        print(partial_traces)

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, Q_m_P_op))
    L_op_tt_adj = tt_rank_reduce(tt_add(L_op_tt_adj, Q_m_P_op_adj))
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, Q_m_P_op_bias))

    # ---
    # VIII
    DS_op = tt_DS_op(n, 2*n)
    DS_op_adj = tt_DS_op_adj(n, 2*n)
    DS_op_bias = tt_DS_bias(n, 2*n)

    def test_DS_op():
        random_A = tt_random_gaussian([3] * (2 * n), shape=(2, 2))
        M = tt_matrix_to_matrix(random_A)
        print(np.round(M, decimals=4))
        print(np.round(tt_matrix_to_matrix(tt_mat(tt_matrix_vec_mul(DS_op, tt_vec(random_A)))), decimals=4))
        m = 2**(2*n)
        P = M[m, :m].reshape(2**n, 2**n)
        print(np.sum(P, axis=0))
        print(np.sum(P, axis=1))

    def test_DS_op_adj():
        random_A = tt_random_gaussian([3] * (2 * n), shape=(2, 2))
        M = tt_matrix_to_matrix(random_A)
        print(np.round(M, decimals=4))
        print(np.round(tt_matrix_to_matrix(tt_mat(tt_matrix_vec_mul(DS_op_adj, tt_vec(random_A)))), decimals=4))
        m = 2**(2*n)
        P_1 = np.concatenate((M[m, :m:2], M[m, :m:2]), axis=0)
        P_2 = np.vstack((np.diagonal(M[:m, :m][-2**n:, -2**n:]), np.diagonal(M[:m, :m][-2**n:, -2**n:]))).flatten(order="F")
        print(P_1 + P_2)

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, DS_op))
    L_op_tt_adj = tt_rank_reduce(tt_add(L_op_tt_adj, DS_op_adj))
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, DS_op_bias))

    # ---
    # IX
    padding_op = tt_padding_op(2*n)
    padding_op_adj = tt_padding_op_adj(2*n)
    padding_op_bias = [np.array([[0.0, 0.0], [0.0, 1.0]]).reshape(1, 2, 2, 1)] + tt_identity(2*n)

    def test_padding_op():
        random_A = tt_random_gaussian([3] * (2 * n), shape=(2, 2))
        M = tt_matrix_to_matrix(random_A)
        print(np.round(M, decimals=4))
        print(np.round(tt_matrix_to_matrix(tt_mat(tt_matrix_vec_mul(padding_op, tt_vec(random_A)))), decimals=4))

    # FIXME: Block-AMeN not happy with this, Do we even need this? We initialise it and it stays psd
    # FIXME: It is too many zeros as before, we get degenerate galerkin projections
    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, padding_op))
    L_op_tt_adj = tt_rank_reduce(tt_add(L_op_tt_adj, padding_op_adj))
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, padding_op_bias))


    # ---
    # Inequality Operator
    # X
    Q_ineq_op = tt_ineq_op(2*n)
    Q_ineq_op_adj = tt_ineq_op_adj(2*n)
    Q_ineq_bias = [(1e-2)*np.ones((1, 2, 2, 1))] + tt_one_matrix(2 * n)

    def test_Q_ineq_op():
        random_A = tt_random_gaussian([3] * (2 * n), shape=(2, 2))
        M = tt_matrix_to_matrix(random_A)
        print(np.round(M, decimals=4))
        print(np.round(tt_matrix_to_matrix(tt_mat(tt_matrix_vec_mul(Q_ineq_op, tt_vec(random_A)))), decimals=4))


    # ---
    print("...Problem created!")
    print(f"Objective TT-ranks: {tt_ranks(C_tt)}")
    print(f"Eq Op-rank: {tt_ranks(L_op_tt)}")
    print(f"Eq Op-adjoint-rank: {tt_ranks(L_op_tt_adj)}")
    print(f"Eq Bias-rank: {tt_ranks(eq_bias_tt)}")
    print("-----------------------------------")
    print(f"Ineq Op-rank: {tt_ranks(Q_ineq_op)}")
    print(f"Ineq Op-adjoint-rank: {tt_ranks(Q_ineq_op_adj)}")
    print(f"Ineq Bias-rank: {tt_ranks(Q_ineq_bias)}")
    t0 = time.time()
    X_tt, Y_tt, T_tt, Z_tt = tt_ipm(
        C_tt,
        L_op_tt,
        L_op_tt_adj,
        eq_bias_tt,
        Q_ineq_op,
        Q_ineq_op_adj,
        Q_ineq_bias,
        max_iter=2,
        verbose=True
    )
    t1 = time.time()
    print("Solution: ")
    print(np.round(tt_matrix_to_matrix(X_tt), decimals=2))
    print(f"Objective value: {tt_inner_prod(C_tt, X_tt)}")
    print("Complementary Slackness: ", tt_inner_prod(X_tt, Z_tt))
    print(f"Ranks X_tt: {tt_ranks(X_tt)}, Y_tt: {tt_ranks(Y_tt)}, \n "
          f"     T_tt: {tt_ranks(T_tt)}, Z_tt: {tt_ranks(Z_tt)} ")
    print(f"Time: {t1 - t0}s")