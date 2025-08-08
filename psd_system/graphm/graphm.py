import sys
import os

sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
from src.utils import run_experiment

Q_PREFIX = [np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1), np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)]


# Constraint 4 -----------------------------------------------------------------

def tt_partial_trace_op(block_size, dim):
    # 4.9
    op_tt = tt_diag(tt_split_bonds(tt_sub(tt_one_matrix(dim - block_size), tt_identity(dim - block_size))))
    block_op = tt_diag(tt_split_bonds(tt_identity(block_size)))
    return tt_reshape(tt_rank_reduce(Q_PREFIX + op_tt + block_op), (4, 4))

# ------------------------------------------------------------------------------
# Constraint 5 -----------------------------------------------------------------

def tt_partial_J_trace_op(block_size, dim):
    #4.11
    matrix_tt = tt_sub(tt_identity(dim - block_size), [E(0, 0) for _  in range(dim-block_size)])
    block_op_0 = []
    for i, c in enumerate(tt_split_bonds(tt_identity(block_size))):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, 1] = c
        block_op_0.append(core)
    op_tt_0 = tt_diag(tt_split_bonds(matrix_tt)) + block_op_0
    # 4.10.1
    matrix_tt = tt_sub(tt_triu_one_matrix(dim-block_size), tt_identity(dim-block_size))
    block_op_1 = []
    for i, c in enumerate(tt_split_bonds(tt_one_matrix(block_size))): # TODO: Should be tt_sub(tt_one_matrix(block_size), tt_identity(block_size)) but this results in lower rank
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, (i+1) % 2] = c
        block_op_1.append(core)
    op_tt_1 = tt_diag(tt_split_bonds(matrix_tt)) + block_op_1
    # 4.10.2
    matrix_tt = tt_sub(tt_tril_one_matrix(dim - block_size), tt_identity(dim - block_size))
    block_op_2 = []
    for i, c in enumerate(tt_split_bonds(tt_one_matrix(block_size))): # TODO: Should be tt_sub(tt_one_matrix(block_size), tt_identity(block_size)) but this results in lower rank
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, i % 2] = c
        block_op_2.append(core)
    op_tt_2 = tt_diag(tt_split_bonds(matrix_tt)) + block_op_2
    return tt_reshape(tt_rank_reduce(Q_PREFIX + tt_sum(op_tt_0, op_tt_1, op_tt_2)), (4, 4))

# ------------------------------------------------------------------------------
# Constraint 6 -----------------------------------------------------------------

def tt_diag_block_sum_linear_op(block_size, dim):
    # 4.12
    op_tt = []
    for c in tt_split_bonds(tt_identity(dim - block_size)):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, 0] = c
        op_tt.append(core)
    block_matrix = tt_identity(block_size)
    op_tt = op_tt + tt_diag(tt_split_bonds(block_matrix))
    # 4.13
    op_tt_2 = tt_diag(tt_split_bonds(tt_identity(dim - block_size)))
    block_matrix = tt_diag(tt_split_bonds(tt_sub(tt_one_matrix(block_size), tt_identity(block_size))))
    op_tt_2 = op_tt_2 + block_matrix

    return tt_reshape(tt_rank_reduce(Q_PREFIX + tt_add(op_tt, op_tt_2)), (4, 4))

# ------------------------------------------------------------------------------
# Constraint 7 -----------------------------------------------------------------

def tt_Q_m_P_op(dim):
    #4.14 |
    Q_part = [E(0,  0), E(1,  0)]
    for i in range(dim):
        core_1 = np.concatenate((E(0, 0), E(1, 1)), axis=-1)
        core_2 = np.concatenate((E(0, 0), E(0, 1)), axis=0)
        Q_part.extend([core_1, core_2])
    P_part = [-E(0, 0), E(1, 1)] + tt_diag(tt_split_bonds([E(0, 0) + E(1, 0) for _ in range(dim)]))
    part_1 = tt_add(Q_part, P_part)
    # --
    Q_part_2 = [E(1, 0), E(0, 0)]
    for i in range(dim):
        core_1 = np.concatenate((E(0, 0), E(0, 1)), axis=-1)
        core_2 = np.concatenate((E(0, 0), E(1, 1)), axis=0)
        Q_part_2.extend([core_1, core_2])
    P_part_2 = [-E(1, 1), E(0, 0)] + tt_diag(tt_split_bonds([E(0, 0) + E(0, 1) for _ in range(dim)]))
    part_2 = tt_add(Q_part_2, P_part_2)
    return tt_reshape(tt_add(part_2, part_1), (4, 4))

# ------------------------------------------------------------------------------
# Constraint 8 -----------------------------------------------------------------

# DS constraint implied by constraint collective of 5, 6, 8

# ------------------------------------------------------------------------------
# Constraint 9 -----------------------------------------------------------------

def tt_padding_op(dim):
    matrix_tt = [E(0, 1) + E(1, 0) +  E(1, 1)] + tt_one_matrix(dim)
    matrix_tt  = tt_sub(matrix_tt, [E(0, 1)] + [E(0, 0) + E(1, 0) for _ in range(dim)])
    matrix_tt = tt_sub(matrix_tt, [E(1, 0)] + [E(0, 0) + E(0, 1) for _ in range(dim)])
    basis = tt_diag(tt_split_bonds(matrix_tt))
    return tt_reshape(tt_rank_reduce(basis), (4, 4))

# ------------------------------------------------------------------------------


def tt_obj_matrix(rank, dim):
    G_A = tt_random_graph(dim, rank)
    print("Graph A: ")
    print(np.round(tt_matrix_to_matrix(G_A), decimals=2))

    G_B = tt_random_graph(dim, rank)
    print("Graph B: ")
    print(np.round(tt_matrix_to_matrix(G_B), decimals=2))

    C_tt = [E(0, 0)] + G_B + G_A
    return C_tt

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


        [ 6  6  | 4  0  | 7 | 0 0 0]
        [ 6  6  | 5  4  | 7 | 0 0 0]
        [--------------------------]
        [ 4  5  | 0  6  | 7 | 0 0 0]
        [ 0  4  | 6  5  | 7 | 0 0 0]
    Y = [--------------------------]
        [ 7  7  | 7  7  | P | 0 0 0]
        [--------------------------]
        [ 0  0  | 0  0  | 0 | P 0 0]
        [ 0  0  | 0  0  | 0 | 0 P 0]
        [ 0  0  | 0  0  | 0 | 0 0 P] 

        8r and 8c implied by other constraints
"""

def create_problem(n, max_rank):
    print("Creating Problem...")

    C_tt = tt_obj_matrix(max_rank, n)

    # Equality Operator
    # IV
    partial_tr_op = tt_partial_trace_op(n, 2 * n)
    #partial_tr_op_bias = tt_zero_matrix(2 * n + 1)

    L_op_tt = partial_tr_op
    #eq_bias_tt = partial_tr_op_bias
    # ---
    # V
    partial_tr_J_op = tt_partial_J_trace_op(n, 2 * n)
    partial_tr_J_op_bias = [E(0, 0)] + tt_sub(tt_tril_one_matrix(n), tt_identity(n)) + [E(0, 1) for _ in range(n)]
    partial_tr_J_op_bias = tt_add(partial_tr_J_op_bias, [E(0, 0)] + tt_sub(tt_triu_one_matrix(n), tt_identity(n)) + [E(1, 0) for _ in range(n)])
    partial_tr_J_op_bias = tt_rank_reduce(tt_add(partial_tr_J_op_bias, [E(0, 0)] + tt_sub(tt_identity(n), [E(0, 0) for _ in range(n)]) + [E(1, 1) for _ in range(n)]))

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, partial_tr_J_op), 1e-12)
    eq_bias_tt = partial_tr_J_op_bias # tt_rank_reduce(tt_add(eq_bias_tt, partial_tr_J_op_bias))

    # ---
    # VI
    diag_block_sum_op = tt_diag_block_sum_linear_op(n, 2 * n)
    diag_block_sum_op_bias = [E(0, 0) for _ in range(n + 1)] + tt_identity(n)

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, diag_block_sum_op), 1e-12)
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, diag_block_sum_op_bias))

    # ---
    # VII
    Q_m_P_op = tt_Q_m_P_op(2 * n)
    #Q_m_P_op_bias = tt_zero_matrix(2 * n + 1)

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, Q_m_P_op), 1e-12)
    #eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, Q_m_P_op_bias))

    # ---
    # Inequality Operator
    # X
    ineq_mask = tt_rank_reduce([E(0, 0)] + tt_sub(tt_one_matrix(n), tt_identity(n)) + tt_sub(tt_one_matrix(n), tt_identity(n)))
    # ---

    # ---
    pad = [1 - E(0, 0)] + tt_one_matrix(2 * n)
    pad = tt_sub(pad, [E(0, 1)] + [E(0, 0) + E(1, 0) for _ in range(2 * n)])
    pad = tt_sub(pad, [E(1, 0)] + [E(0, 0) + E(0, 1) for _ in range(2 * n)])

    lag_map_y = tt_sub(
            tt_one_matrix(2 * n + 1),
            tt_sum(
                pad,  # P
                [E(0, 1)] + [E(0, 0) + E(1, 0) for _ in range(2 * n)],  # 7
                [E(1, 0)] + [E(0, 0) + E(0, 1) for _ in range(2 * n)],  # 7
                [E(0, 0)] + [E(0, 0) for _ in range(n)] + tt_identity(n),
                # 6.1
                [E(0, 0)] + tt_identity(n) + tt_sub(tt_one_matrix(n), tt_identity(n)),  # 6.2
                partial_tr_J_op_bias,  # 5
                [E(0, 0)] + tt_sub(tt_one_matrix(n), tt_identity(n)) + tt_identity(n)  # 4

            )
    )
    lag_map_t = tt_sub(tt_one_matrix(2 * n + 1), ineq_mask)

    lag_maps = {
        "y": tt_diag_op(lag_map_y),
        "t": tt_diag_op(lag_map_t)
    }

    scale = max(2**(2*n + 1 - 7), 1)
    eq_bias_tt = tt_normalise(eq_bias_tt, radius=scale)

    # IX
    padding_op = tt_padding_op(2 * n)
    padding_op_bias = [E(1, 1)] + tt_identity(2 * n)

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, padding_op), 1e-12)
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, padding_op_bias))
    
    return tt_normalise(C_tt, radius=scale), L_op_tt, eq_bias_tt, ineq_mask, lag_maps

if __name__ == "__main__":
    run_experiment(create_problem)