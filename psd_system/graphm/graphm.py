import sys
import os
import yaml
import argparse

sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
from src.tt_ipm import tt_ipm
import time
from memory_profiler import memory_usage


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
    for i, c in enumerate(tt_split_bonds(tt_sub(tt_one_matrix(block_size), tt_identity(block_size)))):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, (i+1) % 2] = c
        block_op_1.append(core)
    op_tt_1 = tt_diag(tt_split_bonds(matrix_tt)) + block_op_1
    # 4.10.2
    matrix_tt = tt_sub(tt_tril_one_matrix(dim - block_size), tt_identity(dim - block_size))
    block_op_2 = []
    for i, c in enumerate(tt_split_bonds(tt_sub(tt_one_matrix(block_size), tt_identity(block_size)))):
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
    #4.14
    Q_part = [E(0,  0), E(1,  0)]
    for i in range(dim):
        core_1 = np.concatenate((E(0, 0), E(1, 1)), axis=-1)
        core_2 = np.concatenate((E(0,0), E(0,  1)), axis=0)
        Q_part.extend([core_1, core_2])
    P_part = [-E(0, 0), E(1, 1)] + tt_diag(tt_split_bonds([E(0, 0) + E(1, 0) for _ in range(dim)]))
    part_1 = tt_add(Q_part, P_part)
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
    matrix_tt = tt_sub(matrix_tt, [E(1,  0)] + [E(0, 0) + E(0, 1) for _ in range(dim)])
    basis = tt_diag(tt_split_bonds(matrix_tt))
    return tt_reshape(tt_rank_reduce(basis), (4, 4))

# ------------------------------------------------------------------------------


def tt_obj_matrix(rank, dim):
    scale = 2 ** (7 - 2*dim)
    G_A = tt_random_graph(dim, rank)
    # print("Graph A: ")
    # print(np.round(tt_matrix_to_matrix(G_A), decimals=2))

    G_B = tt_random_graph(dim, rank)
    # print("Graph B: ")
    # print(np.round(tt_matrix_to_matrix(G_B), decimals=2))

    # print("Objective matrix: ")
    C_tt = [E(0, 0)] + G_B + G_A
    # print(np.round(tt_matrix_to_matrix(C_tt), decimals=2))
    return tt_normalise(C_tt, radius=scale)

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
    partial_tr_op_bias = tt_zero_matrix(2 * n + 1)

    L_op_tt = partial_tr_op
    eq_bias_tt = partial_tr_op_bias
    # ---
    # V
    partial_tr_J_op = tt_partial_J_trace_op(n, 2 * n)
    partial_tr_J_op_bias = [E(0, 0)] + tt_sub(tt_tril_one_matrix(n), tt_identity(n)) + [E(0, 1) for _ in range(n)]
    partial_tr_J_op_bias = tt_add(partial_tr_J_op_bias, [E(0, 0)] + tt_sub(tt_triu_one_matrix(n), tt_identity(n)) + [E(1, 0) for _ in range(n)])
    partial_tr_J_op_bias = tt_rank_reduce(tt_add(partial_tr_J_op_bias, [E(0, 0)] + tt_sub(tt_identity(n), [E(0, 0) for _ in range(n)]) + [E(1, 1) for _ in range(n)]))

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

    return C_tt, L_op_tt, tt_normalise(eq_bias_tt, radius=1), ineq_mask, lag_maps

if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument('--rank', type=int, required=False, help='An integer input', default=0)
    args = parser.parse_args()
    with open(os.getcwd() + '/../../' + args.config, "r") as file:
        config = yaml.safe_load(file)

    problem_creation_times = []
    runtimes = []
    memory = []
    complementary_slackness = []
    feasibility_errors = []
    num_iters = []
    for seed in config["seeds"]:
        print("Seed: ", seed)
        np.random.seed(seed)
        t0 = time.time()
        rank = config["max_rank"] if args.rank == 0 else args.rank
        t1 = time.time()
        C_tt, L_op_tt, eq_bias_tt, ineq_mask, lag_maps = create_problem(config["dim"], config["max_rank"])
        lag_maps = {key: tt_reshape(value, (4, 4)) for key, value in lag_maps.items()}
        C_tt = tt_reshape(C_tt, (4,))
        eq_bias_tt = tt_reshape(eq_bias_tt, (4,))
        t2 = time.time()
        if args.track_mem:
            start_mem = memory_usage(max_usage=True)
            def wrapper():
                X_tt, Y_tt, T_tt, Z_tt, info = tt_ipm(
                    lag_maps,
                    C_tt,
                    L_op_tt,
                    eq_bias_tt,
                    ineq_mask,
                    max_iter=config["max_iter"],
                    verbose=config["verbose"],
                    gap_tol=config["gap_tol"],
                    op_tol=config["op_tol"],
                    warm_up=config["warm_up"],
                    aho_direction=False
                )
                return X_tt, Y_tt, T_tt, Z_tt, info

            res = memory_usage(proc=wrapper, max_usage=True, retval=True)
            X_tt, Y_tt, T_tt, Z_tt, info = res[1]
            memory.append(res[0] - start_mem)
        else:
            X_tt, Y_tt, T_tt, Z_tt, info = tt_ipm(
                lag_maps,
                C_tt,
                L_op_tt,
                eq_bias_tt,
                ineq_mask,
                max_iter=config["max_iter"],
                verbose=config["verbose"],
                gap_tol=config["gap_tol"],
                op_tol=config["op_tol"],
                warm_up=config["warm_up"],
                aho_direction=False
            )

        t3 = time.time()
        problem_creation_times.append(t2 - t1)
        runtimes.append(t3 - t2)
        complementary_slackness.append(abs(tt_inner_prod(X_tt, Z_tt)))
        primal_res = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(L_op_tt, tt_reshape(X_tt, (4,))), eq_bias_tt), eps=1e-12)
        feasibility_errors.append(tt_inner_prod(primal_res, primal_res))
        num_iters.append(info["num_iters"])
        print(f"Converged after {num_iters[-1]:.1f} iterations", flush=True)
        print(f"Problem created in {problem_creation_times[-1]:.3f}s", flush=True)
        print(f"Problem solved in {runtimes[-1]:.3f}s", flush=True)
        if args.track_mem:
            print(f"Peak memory {memory[-1]:.3f} MB", flush=True)
        print(f"Complementary Slackness: {complementary_slackness[-1]}", flush=True)
        print(f"Total feasibility error: {feasibility_errors[-1]}", flush=True)
    print("--- Run Summary ---", flush=True)
    print(f"Converged after avg {np.mean(num_iters):.1f} iterations", flush=True)
    print(f"Problem created in avg {np.mean(problem_creation_times):.3f}s", flush=True)
    print(f"Problem solved in avg {np.mean(runtimes):.3f}s", flush=True)
    if args.track_mem:
        print(f"Peak memory avg {np.mean(memory):.3f} MB", flush=True)
    print(f"Complementary Slackness avg: {np.mean(complementary_slackness)}", flush=True)
    print(f"Total feasibility error avg: {np.mean(feasibility_errors)}", flush=True)
