import copy
import sys
import os
import time

sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from psd_system.graph_plotting import *


@dataclass
class Config:
    seed = 99
    ranks = [5, 5, 5]


if __name__ == "__main__":
    print("Creating Problem...")

    np.random.seed(Config.seed)
    graph = tt_random_graph(Config.ranks)
    G = tt_scale(0.5, tt_add(graph, tt_one(len(Config.ranks) + 1, shape=(2, 2))))
    G = tt_rank_reduce(G)
    print(np.round(tt_op_to_matrix(G), decimals=2))

    As = tt_mask_to_linear_op(G)
    bias = [np.zeros((1, 4, 1)) for _ in range(len(G))]
    J = tt_scale(-1, [np.ones((1, 2, 2, 1)) for _ in range(len(G))])
    #k = tt_eval_constraints(As, J)
    #k = [c.reshape(c.shape[0], 2, 2, c.shape[-1]) for c in k]
    #ones = [np.ones((1, 4, 1)) for _ in range(len(G))]
    #s = tt_constraint_contract(As, ones)
    #print(np.round(tt_op_to_matrix(s), decimals=2))

    print("...Problem created!")
    print(f"Objective Ranks: {tt_ranks(J)}")
    print(f"Constraint Ranks: As {tt_ranks(As)}, bias {tt_ranks(bias)}")
    t0 = time.time()
    X, duality_gaps = tt_sdp_fw(J, As, bias, trace_param_root_n=(1, 1), num_iter=200)
    t1 = time.time()
    print(f"Problem solved in {t1 - t0:.3f}s")
    print(f"Objective value: {tt_inner_prod(tt_scale(-1, J), X)}")
    evaled_constraints = tt_linear_op(As, X)
    scaled_error = [c / np.sqrt(c.shape[1]) for c in tt_add(evaled_constraints, tt_scale(-1, bias))]
    avg_error = np.sqrt(tt_inner_prod(scaled_error, scaled_error))
    print(f"Avg constraint error: {avg_error}")
    print("Ranks of X: ", tt_ranks(X))
    solution = tt_op_to_matrix(X)
    print(np.round(solution, decimals=2))
    plot_duality_gaps(duality_gaps)
