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
    seed = 9 #999: Very low rank solution, 9: Low rank solution, 3: Regular solution
    ranks = [3]


if __name__ == "__main__":
    np.random.seed(Config.seed)
    print("Creating Problem...")
    G_tt = tt_rank_reduce(tt_random_graph(Config.ranks)) # [np.array([[-1, 1], [1, -1]]).reshape(1, 2, 2, 1)] + [np.ones((1, 2, 2, 1)) for _ in Config.ranks]#
    print(np.round(tt_matrix_to_matrix(G_tt), decimals=2))
    As_tt = tt_mask_to_linear_op(tt_identity(len(G_tt)))
    bias_tt = tt_identity(len(G_tt))
    print("...Problem created!")
    print(f"Objective Ranks: {tt_ranks(G_tt)}")
    print(f"Constraint Ranks: As {tt_ranks(As_tt)}, bias {tt_ranks(bias_tt)}")
    t0 = time.time()
    X_tt, Y_tt, Z_tt = tt_ipm(G_tt, As_tt, bias_tt, verbose=True)
    t1 = time.time()
    print("Solution: ")
    np.set_printoptions(linewidth=600, threshold=np.inf, precision=4, suppress=True)
    print(np.round(tt_matrix_to_matrix(X_tt), decimals=2))
    print(f"Objective value: {tt_inner_prod(G_tt, X_tt)}")
    print("Complementary Slackness: ", tt_inner_prod(X_tt, Z_tt))
    print(f"Ranks X_tt: {tt_ranks(X_tt)}, Y_tt: {tt_ranks(Y_tt)}, Z_tt: {tt_ranks(Z_tt)} ")
    print(f"Time: {t1-t0}s")
    """
    print(f"Problem solved in {t1 - t0:.3f}s")
    print(f"Objective value: {-tt_inner_prod(G, X)}")
    evaled_constraints = tt_linear_op(As, X)
    scaled_error = [c / np.sqrt(c.shape[1]) for c in tt_add(evaled_constraints, tt_scale(-1, bias))]
    avg_error = np.sqrt(tt_inner_prod(scaled_error, scaled_error))
    print(f"Avg constraint error: {avg_error}")
    print("Ranks of X: ", tt_ranks(X))
    """