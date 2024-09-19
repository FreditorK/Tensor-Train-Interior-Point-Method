import sys
import os
import time

sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from psd_system.graph_plotting import *
from src.tt_ipm import tt_ipm, _tt_get_block



@dataclass
class Config:
    seed = 3
    ranks = [3]


if __name__ == "__main__":
    np.random.seed(Config.seed)
    print("Creating Problem...")
    G = tt_random_graph(Config.ranks)
    print(np.round(tt_matrix_to_matrix(G), decimals=2))
    As = tt_mask_to_linear_op(tt_identity(len(G)))
    bias = tt_identity(len(G))
    print("...Problem created!")
    print(f"Objective Ranks: {tt_ranks(G)}")
    print(f"Constraint Ranks: As {tt_ranks(As)}, bias {tt_ranks(bias)}")
    t0 = time.time()
    V_tt, Y_tt = tt_ipm(G, As, bias, max_iter=10)
    t1 = time.time()
    VX_tt = tt_rank_reduce(_tt_get_block(0, 0, V_tt))
    print("V_tt-rank: ", tt_ranks(V_tt), tt_ranks(VX_tt))
    XZ_tt = tt_rank_reduce(tt_mat_mat_mul(V_tt, tt_transpose(V_tt)))
    X_tt = tt_rank_reduce(_tt_get_block(0, 0, XZ_tt))
    Z_tt = tt_rank_reduce(_tt_get_block(1, 1, XZ_tt))
    print("XZ_tt-rank: ", tt_ranks(XZ_tt), tt_ranks(X_tt))
    print(np.round(tt_matrix_to_matrix(XZ_tt), decimals=2))
    print("Complementary Slackness: ", tt_inner_prod(X_tt, Z_tt))
    print(f"Objective value: {tt_inner_prod(G, X_tt)}")
    """
    print(f"Problem solved in {t1 - t0:.3f}s")
    print(f"Objective value: {-tt_inner_prod(G, X)}")
    evaled_constraints = tt_linear_op(As, X)
    scaled_error = [c / np.sqrt(c.shape[1]) for c in tt_add(evaled_constraints, tt_scale(-1, bias))]
    avg_error = np.sqrt(tt_inner_prod(scaled_error, scaled_error))
    print(f"Avg constraint error: {avg_error}")
    print("Ranks of X: ", tt_ranks(X))
    """