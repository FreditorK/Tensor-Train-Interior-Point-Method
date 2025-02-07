import copy
import sys
import os

import numpy as np


sys.path.append(os.getcwd() + '/../../')

from graph_matching import *
from src.regular_ipm import ipm
import time

if __name__ == "__main__":
    n = Config.n
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    print("Creating Problem...")
    np.random.seed(Config.seed)
    G_A = tt_random_graph(n, Config.max_rank)
    print("Graph A: ")
    print(np.round(tt_matrix_to_matrix(G_A), decimals=2))

    G_B = tt_random_graph(n, Config.max_rank)
    print("Graph B: ")
    print(np.round(tt_matrix_to_matrix(G_B), decimals=2))

    print("Objective matrix: ")
    C_tt = [-E(0, 0)] + tt_kron(G_B, G_A)
    print(np.round(tt_matrix_to_matrix(tt_kron(G_B, G_A)), decimals=2))

    # Equality Operator
    # IV
    partial_tr_op = tt_partial_trace_op(n, 2*n)
    partial_tr_op_adj = tt_transpose(partial_tr_op)
    partial_tr_op_bias = tt_zero_matrix(2 * n + 1)

    L_op_tt = partial_tr_op
    L_op_tt_adj = partial_tr_op_adj
    eq_bias_tt = partial_tr_op_bias
    # ---
    # V
    partial_tr_J_op = tt_partial_J_trace_op(n, 2*n)
    partial_tr_J_op_adj = tt_transpose(partial_tr_J_op)
    partial_tr_J_op_bias = ([E(0,  0)]
                            + tt_sub(tt_one_matrix(n), [E(0, 0) for _ in range(n)] )
                            + [E(1, 1) for _ in range(n)])

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, partial_tr_J_op))
    L_op_tt_adj = tt_rank_reduce(tt_add(L_op_tt_adj, partial_tr_J_op_adj))
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, partial_tr_J_op_bias))

    # ---
    # VI
    diag_block_sum_op = tt_diag_block_sum_linear_op(n, 2*n)
    diag_block_sum_op_adj = tt_transpose(diag_block_sum_op)
    diag_block_sum_op_bias = [E(0, 0) for _ in range(n+1)] + tt_identity(n)

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, diag_block_sum_op))
    L_op_tt_adj = tt_rank_reduce(tt_add(L_op_tt_adj, diag_block_sum_op_adj))
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, diag_block_sum_op_bias))

    # ---
    # VII
    Q_m_P_op = tt_Q_m_P_op(2*n)
    Q_m_P_op_adj = tt_transpose(Q_m_P_op)
    Q_m_P_op_bias = tt_zero_matrix(2*n + 1)

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, Q_m_P_op))
    L_op_tt_adj = tt_rank_reduce(tt_add(L_op_tt_adj, Q_m_P_op_adj))
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, Q_m_P_op_bias))

    # ---
    # IX
    padding_op = tt_padding_op(2*n)
    padding_op_adj = tt_transpose(padding_op)
    padding_op_bias = [E(1, 1)] + tt_identity(2*n)

    L_op_tt = tt_rank_reduce(tt_add(L_op_tt, padding_op))
    L_op_tt_adj = tt_rank_reduce(tt_add(L_op_tt_adj, padding_op_adj))
    eq_bias_tt = tt_rank_reduce(tt_add(eq_bias_tt, padding_op_bias))

    # ---
    # Inequality Operator
    # X
    Q_ineq_op = tt_ineq_op(2*n)
    Q_ineq_op_adj = tt_ineq_op_adj(2*n)
    Q_ineq_bias = tt_rank_reduce(tt_scale(0.02, tt_mat(tt_matrix_vec_mul(Q_ineq_op_adj, [np.ones((1, 2, 1)) for _ in range(2*(2*n+1))]))))

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
                    [E(0, 0)] + [E(0,0) for _ in range(n)] + tt_identity(n),
                    # 6.1
                    [E(0, 0)] + [E(0,  0) for _ in range(n)] + [E(1, 0) for _ in range(n)],  # 6.2
                    [E(0,  0)] + tt_sub(
                        tt_one_matrix(n) + [E(1, 1) for _ in range(n)],
                        [E(0, 0) for _ in range(n)] + [E(1, 1) for _ in range(n)]),  # 5
                    [E(0, 0)] + [E(1, 0) for _ in range(n)] + [E(0, 0) for _ in range(n)]  # 4

                )
            )
        ))),
        "t": tt_rank_reduce(tt_diag(tt_vec([E(0, 1) + E(1,  0) + E(1, 1)] + tt_one_matrix(2*n))))
    }


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
    X, Y, T, Z = ipm(
        lag_maps,
        C_tt,
        L_op_tt,
        eq_bias_tt,
        Q_ineq_op,
        Q_ineq_bias,
        max_iter=18,
        verbose=True
    )
    t1 = time.time()
    print("Solution: ")
    print(np.round(X, decimals=2))
    print(f"Objective value: {np.trace(tt_matrix_to_matrix(C_tt).T @ X)}")
    print("Complementary Slackness: ", np.trace(X.T @ Z))
    print(f"Time: {t1 - t0}s")
    # TODO: Something went wrong when generalising to n=2

