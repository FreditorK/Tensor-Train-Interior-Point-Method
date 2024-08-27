import sys
import os
import time

sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from psd_system.graph_plotting import *
from src.tt_ipm import tt_ipm



@dataclass
class Config:
    seed = 9
    ranks = [5]


if __name__ == "__main__":
    np.random.seed(Config.seed)
    print("Creating Problem...")
    G = tt_random_graph(Config.ranks)
    G = tt_scale(-1, G)
    As = tt_mask_to_linear_op(tt_identity(len(G)))
    bias = tt_identity(len(G))
    print("...Problem created!")
    print(f"Objective Ranks: {tt_ranks(G)}")
    print(f"Constraint Ranks: As {tt_ranks(As)}, bias {tt_ranks(bias)}")
    t0 = time.time()
    X = tt_ipm(G, As, bias)
    t1 = time.time()
    """
    print(f"Problem solved in {t1 - t0:.3f}s")
    print(f"Objective value: {-tt_inner_prod(G, X)}")
    evaled_constraints = tt_linear_op(As, X)
    scaled_error = [c / np.sqrt(c.shape[1]) for c in tt_add(evaled_constraints, tt_scale(-1, bias))]
    avg_error = np.sqrt(tt_inner_prod(scaled_error, scaled_error))
    print(f"Avg constraint error: {avg_error}")
    print("Ranks of X: ", tt_ranks(X))
    """