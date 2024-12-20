import sys
import os
import time

sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from src.tt_ipm import tt_ipm



@dataclass
class Config:
    seed = 3 #999: Very low rank solution, 9: Low rank solution, 3: Regular solution
    max_rank = 2
    dim = 4


def tt_diag_op(dim):
    identity = tt_identity(dim)
    basis = tt_diag(tt_vec(identity))
    return basis

def tt_diag_op_adj(dim):
    return tt_diag_op(dim)


if __name__ == "__main__":
    np.random.seed(Config.seed)
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    print("Creating Problem...")
    G_tt = tt_rank_reduce(tt_random_graph(Config.dim, Config.max_rank))
    #G_tt = tt_sub(tt_scale(2, G_tt), tt_one_matrix(Config.dim))
    print(np.round(tt_matrix_to_matrix(G_tt), decimals=2))
    diag_tt_op = tt_diag_op(Config.dim)
    diag_tt_op_adjoint = tt_diag_op_adj(Config.dim)
    bias_tt = tt_identity(Config.dim)

    print("...Problem created!")
    print(f"Objective Ranks: {tt_ranks(G_tt)}")
    print(f"Constraint Ranks: As {tt_ranks(diag_tt_op)}, bias {tt_ranks(bias_tt)}")
    t0 = time.time()
    X_tt, Y_tt, _, Z_tt = tt_ipm(G_tt, diag_tt_op, diag_tt_op_adjoint, bias_tt, max_iter=15, verbose=True)
    t1 = time.time()
    print("Solution: ")
    print(np.round(tt_matrix_to_matrix(X_tt), decimals=2))
    print(f"Objective value: {tt_inner_prod(G_tt, X_tt)}")
    print("Complementary Slackness: ", tt_inner_prod(X_tt, Z_tt))
    print(f"Ranks X_tt: {tt_ranks(X_tt)}, Y_tt: {tt_ranks(Y_tt)}, Z_tt: {tt_ranks(Z_tt)} ")
    print(f"Time: {t1-t0}s")