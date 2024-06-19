import sys
import os
import time

sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from psd_system.graph_plotting import *


@dataclass
class Config:
    seed = 9
    ranks = [5, 5, 5, 5]


if __name__ == "__main__":
    np.random.seed(Config.seed)
    print("Creating Problem...")
    G = tt_random_graph(Config.ranks)
    G = tt_scale(-1, G)
    diag_core = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]]).reshape(1, 2, 2, 2, 1)
    As = [diag_core] * len(G)
    bias = tt_one(len(G))
    print("...Problem created!")
    print(f"Objective Ranks: {tt_ranks(G)}")
    print(f"Constraint Ranks: As {tt_ranks(As)}, bias {tt_ranks(bias)}")
    t0 = time.time()
    X, duality_gaps = tt_sdp_fw(G, As, bias, trace_param_root_n=(2, 2), num_iter=100)
    t1 = time.time()
    print(f"Problem solved in {t1 - t0:.3f}s")
    print(f"Objective value: {-tt_inner_prod(G, X)}")
    evaled_constraints = tt_eval_constraints(As, X)
    scaled_error = [c / 2 for c in tt_add(evaled_constraints, tt_scale(-1, bias))]
    avg_error = np.sqrt(tt_inner_prod(scaled_error, scaled_error))
    print(f"Avg constraint error: {avg_error}")
    print("Ranks of X: ", tt_ranks(X))
    #tt_eig, eig_val = tt_randomised_max_eigentensor(X, num_iter=100)
    #tt_eig = tt_binary_round(tt_eig)
    #sol_vec = tt_to_tensor(tt_eig).flatten()
    solution = tt_op_to_matrix(X)
    chol = robust_cholesky(solution, epsilon=0.01)
    nodes_in_cut = [i for i, v in enumerate(chol.T @ np.random.randn(chol.shape[0], 1)) if v > 0]
    adj_matrix = -tt_op_to_matrix(G)
    plot_maxcut(adj_matrix, nodes_in_cut, duality_gaps)
