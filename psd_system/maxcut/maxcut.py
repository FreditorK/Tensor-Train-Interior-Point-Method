import sys
import os
import time

sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from src.tt_ipm import tt_ipm
from src.tt_eig import tt_min_eig, tt_max_eig



@dataclass
class Config:
    seed = 3 #999: Very low rank solution, 9: Low rank solution, 3: Regular solution
    max_rank = 2
    dim = 5 #max 9 symmetric op


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
    #print(np.round(tt_matrix_to_matrix(G_tt), decimals=2))
    L_tt = tt_diag_op(Config.dim)
    bias_tt = tt_identity(Config.dim)

    lag_maps = {"y": tt_rank_reduce(tt_diag(tt_vec(tt_sub(tt_one_matrix(Config.dim), tt_identity(Config.dim)))))}

    print("...Problem created!")
    print(f"Objective Ranks: {tt_ranks(G_tt)}")
    print(f"Constraint Ranks: As {tt_ranks(L_tt)}, bias {tt_ranks(bias_tt)}")
    t0 = time.time()
    X_tt, Y_tt, _, Z_tt = tt_ipm(lag_maps, G_tt, L_tt, bias_tt, max_iter=3, verbose=True)
    t1 = time.time()
    print("Solution: ")
    print(np.round(tt_matrix_to_matrix(X_tt), decimals=2))
    print(f"Objective value: {tt_inner_prod(G_tt, X_tt)}")
    print("Complementary Slackness: ", tt_inner_prod(X_tt, Z_tt))
    print(f"Ranks X_tt: {tt_ranks(X_tt)}, Y_tt: {tt_ranks(Y_tt)}, Z_tt: {tt_ranks(Z_tt)} ")
    print(f"Time: {t1-t0}s")