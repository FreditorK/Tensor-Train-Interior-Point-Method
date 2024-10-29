import copy
import sys
import os

from cvxpy.interface import shape

sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from psd_system.graph_plotting import *
from src.tt_ipm import tt_ipm, _tt_get_block
import time


@dataclass
class Config:
    seed = 4
    ranks = [3]


def tt_tr_op(dim):
    tr_tt_op = [np.zeros((1, 4, 2, 2, 1)) for _ in range(dim)]
    for c in tr_tt_op:
        c[:, 0, :, :, 0] = np.eye(2)
    return tr_tt_op

def tt_tr_op_adjoint(dim):
    tr_tt_op = [np.zeros((1, 4, 2, 2, 1)) for _ in range(dim)]
    for c in tr_tt_op:
        c[:, 0, :, :, 0] = np.eye(2)
        c[:, 3, :, :, 0] = np.eye(2)
    return tr_tt_op


if __name__ == "__main__":
    print("Creating Problem...")

    np.random.seed(Config.seed)
    graph = tt_random_graph(Config.ranks)
    G = tt_scale(0.5, tt_add(graph, tt_one_matrix(len(Config.ranks) + 1)))
    G = tt_rank_reduce(G)
    print(np.round(tt_matrix_to_matrix(G), decimals=2))
    n = len(G)

    # I
    As_tt_op = tt_mask_to_linear_op(G)
    As_tt_op_adjoint = tt_mask_to_linear_op_adjoint(G)
    bias_tt = tt_zero_matrix(n)
    # II
    tr_tt_op = tt_tr_op(n)
    tr_tt_op_adjoint = tt_tr_op_adjoint(n)
    tr_bias_tt = [np.array([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(n)]

    # Objective
    J_tt = tt_one_matrix(n)

    # Constraint
    L_tt = tt_rank_reduce(tt_add(As_tt_op, tr_tt_op))
    L_tt_adjoint = tt_rank_reduce(tt_add(As_tt_op_adjoint, tr_tt_op_adjoint))
    bias_tt = tt_rank_reduce(tt_add(bias_tt, tr_bias_tt))

    Q_ineq_op = tt_mask_to_linear_op(tt_one_matrix(n))
    Q_ineq_op_adjoint = tt_mask_to_linear_op_adjoint(tt_one_matrix(n))
    Q_ineq_bias = tt_scale(-1, tt_one_matrix(n))

    print("...Problem created!")
    print(f"Objective Ranks: {tt_ranks(J_tt)}")
    print(f"Constraint Ranks: \n \t As {tt_ranks(L_tt)}, bias {tt_ranks(bias_tt)}")
    t0 = time.time()
    X_tt, Y_tt, _, Z_tt = tt_ipm(
        J_tt,
        L_tt,
        L_tt_adjoint,
        bias_tt,
        Q_ineq_op,
        Q_ineq_op_adjoint,
        Q_ineq_bias,
        verbose=True
    )
    t1 = time.time()
    print("Solution: ")
    print(np.round(tt_matrix_to_matrix(X_tt), decimals=2))
    print(f"Problem solved in {t1 - t0:.3f}s")
    print(f"Objective value: {tt_inner_prod(J_tt, X_tt)}")
    print("Complementary Slackness: ", tt_inner_prod(X_tt, Z_tt))
    print(f"Ranks- X_tt {tt_ranks(X_tt)} Y_tt {tt_ranks(Y_tt)} Z_tt {tt_ranks(Z_tt)} ")
