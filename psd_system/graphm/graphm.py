import sys
import os
import yaml
import tracemalloc
import argparse

sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
from src.tt_ipm import tt_ipm
import time


Q_PREFIX = [np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1), np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1)]


# Constraint 4 -----------------------------------------------------------------

def tt_partial_trace_op(block_size, dim):
    op_tt = []
    for i, c in enumerate(tt_vec(tt_sub(tt_one_matrix(dim - block_size), tt_identity(dim - block_size)))):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, (i+1) % 2] = c
        op_tt.append(core)
    block_op = []
    for i, c in enumerate(tt_vec(tt_identity(block_size))):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, 0] = c
        block_op.append(core)
    return tt_rank_reduce(Q_PREFIX + op_tt + block_op)

# ------------------------------------------------------------------------------
# Constraint 5 -----------------------------------------------------------------

def tt_partial_J_trace_op(block_size, dim):
    matrix_tt = tt_sub(tt_identity(dim - block_size), [E(0, 0) for _  in range(dim-block_size)])
    block_op_0 = []
    for i, c in enumerate(tt_vec(tt_identity(block_size))):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, 1] = c
        block_op_0.append(core)
    op_tt_0 = tt_diag(tt_vec(matrix_tt)) + block_op_0

    matrix_tt = tt_sub(tt_one_matrix(dim-block_size), tt_identity(dim-block_size))
    block_op_1 = []
    for i, c in enumerate(tt_vec(tt_sub(tt_one_matrix(block_size), tt_identity(block_size)))):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, 1] = c
        block_op_1.append(core)
    op_tt_1 = tt_diag(tt_vec(matrix_tt)) + block_op_1
    return tt_rank_reduce(Q_PREFIX + tt_sum(op_tt_0, op_tt_1))

# ------------------------------------------------------------------------------
# Constraint 6 -----------------------------------------------------------------

def tt_diag_block_sum_linear_op(block_size, dim):
    op_tt = []
    for c in tt_vec(tt_identity(dim-block_size)):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, 0, :] = c
        op_tt.append(core)
    block_matrix = tt_identity(block_size)
    op_tt = op_tt + tt_diag(tt_vec(block_matrix))

    op_tt_2 = []
    for c in tt_vec(tt_identity(dim - block_size)):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, 0, :] = c
        op_tt_2.append(core)
    block_matrix = []
    for i, c in enumerate(tt_vec(tt_sub(tt_one_matrix(block_size), tt_identity(block_size)))):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, (i+1) % 2, :] = c
        block_matrix.append(core)
    op_tt_2 = op_tt_2 + block_matrix

    return tt_rank_reduce(Q_PREFIX + tt_add(op_tt, op_tt_2))

# ------------------------------------------------------------------------------
# Constraint 7 -----------------------------------------------------------------

def tt_Q_m_P_op(dim):
    Q_part = [E(0,  0), E(1,  0)]
    for i in range(dim):
        core_1 = np.concatenate((E(0, 0), E(1, 1)), axis=-1)
        core_2 = np.concatenate((E(0,0), E(0,  1)), axis=0)
        Q_part.extend([core_1, core_2])
    P_part = [-0.5*E(0, 0), E(1, 1)] + tt_diag(tt_vec([E(0,  0) + E(1,  0) for _ in range(dim)]))
    P_supplement = [-0.5*E(0, 1), E(1, 0)]
    for i in range(dim):
        core_1 = np.concatenate((E(0, 0), E(1,  0)), axis=-1)
        core_2 = np.concatenate((E(0, 0), E(0,  1)), axis=0)
        P_supplement.extend([core_1, core_2])
    return tt_rank_reduce(tt_add(Q_part, tt_add(P_supplement, P_part)))

# ------------------------------------------------------------------------------
# Constraint 8 -----------------------------------------------------------------

# DS constraint implied by constraint collective of 5, 6, 8

# ------------------------------------------------------------------------------
# Constraint 9 -----------------------------------------------------------------

def tt_padding_op(dim):
    matrix_tt = [E(0, 1) + E(1, 0) +  E(1, 1)] + tt_one_matrix(dim)
    matrix_tt  = tt_sub(matrix_tt, [E(0, 1)] + [E(0, 0) + E(1, 0) for _ in range(dim)])
    matrix_tt = tt_sub(matrix_tt, [E(1,  0)] + [E(0, 0) + E(0, 1) for _ in range(dim)])
    basis = tt_diag(tt_vec(matrix_tt))
    return tt_rank_reduce(basis)

# ------------------------------------------------------------------------------

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


        [ 6  0  | 0  0  | 7 | 0 0 0]
        [ 6  6  | 0  5  | 7 | 0 0 0]
        [--------------------------]
        [ 4  0  | 0  0  | 7 | 0 0 0]
        [ 0  5  | 0  5  | 7 | 0 0 0]
    Y = [--------------------------]
        [ 0  0  | 0  0  | P | 0 0 0]
        [--------------------------]
        [ 0  0  | 0  0  | 0 | P 0 0]
        [ 0  0  | 0  0  | 0 | 0 P 0]
        [ 0  0  | 0  0  | 0 | 0 0 P] 

        8r and 8c implied by other constraints
"""

def create_problem(n, seed, max_rank):
    print("Creating Problem...")
    np.random.seed(seed)
    G_A = tt_random_graph(n, max_rank)
    print("Graph A: ")
    print(np.round(tt_matrix_to_matrix(G_A), decimals=2))

    G_B = tt_random_graph(n, max_rank)
    print("Graph B: ")
    print(np.round(tt_matrix_to_matrix(G_B), decimals=2))

    print("Objective matrix: ")
    C_tt = [-E(0, 0)] + G_B + G_A
    print(np.round(tt_matrix_to_matrix(C_tt), decimals=2))

    # Equality Operator
    # IV
    partial_tr_op = tt_partial_trace_op(n, 2 * n)
    partial_tr_op_bias = tt_zero_matrix(2 * n + 1)

    L_op_tt = partial_tr_op
    eq_bias_tt = partial_tr_op_bias
    # ---
    # V
    partial_tr_J_op = tt_partial_J_trace_op(n, 2 * n)
    partial_tr_J_op_bias = ([E(0, 0)]
                            + tt_sub(tt_one_matrix(n), [E(0, 0) for _ in range(n)])
                            + [E(1, 1) for _ in range(n)])

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, partial_tr_J_op))
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, partial_tr_J_op_bias))

    # ---
    # VI
    diag_block_sum_op = tt_diag_block_sum_linear_op(n, 2 * n)
    diag_block_sum_op_bias = [E(0, 0) for _ in range(n + 1)] + tt_identity(n)

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, diag_block_sum_op))
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, diag_block_sum_op_bias))

    # ---
    # VII
    Q_m_P_op = tt_Q_m_P_op(2 * n)
    Q_m_P_op_bias = tt_zero_matrix(2 * n + 1)

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, Q_m_P_op))
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, Q_m_P_op_bias))

    # ---
    # IX
    padding_op = tt_padding_op(2 * n)
    padding_op_bias = [E(1, 1)] + tt_identity(2 * n)

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, padding_op))
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, padding_op_bias))

    # ---
    # Inequality Operator
    # X
    ineq_mask = [E(0, 0)] + [np.ones((1, 2, 2, 1)) for _ in range(2*n)]

    # ---

    # ---
    pad = [1 - E(0, 0)] + tt_one_matrix(2 * n)
    pad = tt_sub(pad, [E(0, 1)] + [E(0, 0) + E(1, 0) for _ in range(2 * n)])
    pad = tt_sub(pad, [E(1, 0)] + [E(0, 0) + E(0, 1) for _ in range(2 * n)])

    lag_maps = {
        "y": tt_rank_reduce(tt_diag(tt_vec(
            tt_sub(
                tt_one_matrix(2 * n + 1),
                tt_sum(
                    pad,  # P
                    [E(0, 1)] + [E(0, 0) + E(1, 0) for _ in range(2 * n)],  # 7
                    [E(0, 0)] + [E(0, 0) for _ in range(n)] + tt_identity(n),
                    # 6.1
                    [E(0, 0)] + [E(0, 0) for _ in range(n)] + [E(1, 0) for _ in range(n)],  # 6.2
                    [E(0, 0)] + tt_sub(
                        tt_one_matrix(n) + [E(1, 1) for _ in range(n)],
                        [E(0, 0) for _ in range(n)] + [E(1, 1) for _ in range(n)]),  # 5
                    [E(0, 0)] + [E(1, 0) for _ in range(n)] + [E(0, 0) for _ in range(n)]  # 4

                )
            )
        ))),
        "t": tt_rank_reduce(tt_diag(tt_vec(tt_sub(tt_one_matrix(2 * n+1), ineq_mask))))
    }
    return C_tt, L_op_tt, eq_bias_tt, ineq_mask, lag_maps

if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    with open(os.getcwd() + '/../../' + args.config, "r") as file:
        config = yaml.safe_load(file)

    C_tt, L_op_tt, eq_bias_tt, ineq_mask, lag_maps = create_problem(config["dim"], config["seeds"][0], config["max_rank"])
    print("...Problem created!")
    print(f"Objective TT-ranks: {tt_ranks(C_tt)}")
    print(f"Eq Op-rank: {tt_ranks(L_op_tt)}")
    print(f"Eq Bias-rank: {tt_ranks(eq_bias_tt)}")
    if args.track_mem:
        print("Memory tracking started...")
        tracemalloc.start()  # Start memory tracking
    t0 = time.time()
    X_tt, Y_tt, T_tt, Z_tt, info = tt_ipm(
        lag_maps,
        C_tt,
        L_op_tt,
        eq_bias_tt,
        ineq_mask,
        max_iter=config["max_iter"],
        verbose=config["verbose"],
        feasibility_tol=config["feasibility_tol"],
        centrality_tol=config["centrality_tol"],
        op_tol=config["op_tol"]
    )
    t1 = time.time()
    if args.track_mem:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 10 ** 6:.2f} MB")
        print(f"Peak memory usage: {peak / 10 ** 6:.2f} MB")
        tracemalloc.stop()  # Stop tracking after measuring
    print("Solution: ")
    print(np.round(tt_matrix_to_matrix(X_tt), decimals=2))
    print(f"Objective value: {tt_inner_prod(C_tt, X_tt)}")
    print("Complementary Slackness: ", tt_inner_prod(X_tt, Z_tt))
    primal_res = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(L_op_tt, tt_vec(X_tt)), tt_vec(eq_bias_tt)), eps=1e-10)
    print(f"Total primal feasibility error: {np.sqrt(np.abs(tt_inner_prod(primal_res, primal_res)))}")
    print(f"Ranks X_tt: {tt_ranks(X_tt)}, Y_tt: {tt_ranks(Y_tt)}, \n "
          f"     T_tt: {tt_ranks(T_tt)}, Z_tt: {tt_ranks(Z_tt)} ")
    print(f"Time: {t1 - t0}s")
