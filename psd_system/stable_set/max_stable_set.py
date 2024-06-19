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
    seed = 9
    ranks = [5, 5]


if __name__ == "__main__":
    print("Creating Problem...")
    one_frame = [np.zeros((1, 2, 2, 1)) for c in range(len(Config.ranks) + 1)]
    for c in one_frame:
        c[:, 0, :] += 1
    one_frame = tt_add(one_frame, [np.array([[-1, 0], [0, 0]]).reshape(1, 2, 2, 1) for _ in range(len(Config.ranks) + 1)])
    one_frame = tt_rank_reduce(tt_add(one_frame, tt_transpose(one_frame)))
    anti_one_frame = tt_add(tt_one(len(Config.ranks)+1, shape=(2, 2)), tt_scale(-1, one_frame))

    np.random.seed(Config.seed)
    graph = tt_random_graph(Config.ranks)
    G = tt_scale(0.5, tt_add(copy.copy(graph), tt_one(len(Config.ranks) + 1, shape=(2, 2))))
    G_complement = tt_scale(0.5, tt_add(tt_scale(-1, graph), tt_one(len(Config.ranks) + 1, shape=(2, 2))))
    G = tt_rank_reduce(tt_hadamard(anti_one_frame, G))
    G_complement = tt_rank_reduce(tt_hadamard(anti_one_frame, tt_add(G_complement, [np.array([[-1, 0], [0, 0]]).reshape(1, 2, 2, 1) for _ in range(len(Config.ranks) + 1)])))
    #print(np.round(tt_op_to_matrix(G_complement), decimals=2))
    #print(np.round(tt_op_to_matrix(G), decimals=2))

    I_wtho_lead = tt_rank_reduce(tt_add(tt_indentity(len(G_complement)), [np.array([[-1, 0], [0, 0]]).reshape(1, 2, 2, 1) for _ in range(len(G_complement))]))
    G_mask = tt_rank_reduce(tt_add(G_complement, one_frame))

    As = tt_mask_to_linear_op(G_mask)
    bias = [c.reshape(c.shape[0], 4, c.shape[-1]) for c in tt_rank_reduce(tt_add(I_wtho_lead, one_frame))]
    C = [np.array([[1, 0], [0, 0]]).reshape(1, 2, 2, 1) for _ in range(len(G_complement))]

    #k = tt_eval_constraints(As, tt_one(len(Config.ranks) + 1, shape=(2, 2)))
    #print(np.round(tt_op_to_matrix(G_mask), decimals=2))
    #print(np.round(tt_op_to_matrix([c.reshape(c.shape[0], 2, 2, c.shape[-1]) for c in k]), decimals=2))


    print("...Problem created!")
    print(f"Objective Ranks: {tt_ranks(C)}")
    print(f"Constraint Ranks: As {tt_ranks(As)}, bias {tt_ranks(bias)}")
    t0 = time.time()
    X, duality_gaps = tt_sdp_fw(C, As, bias, trace_param_root_n=(2, 2.1), num_iter=200)
    t1 = time.time()
    print(f"Problem solved in {t1 - t0:.3f}s")
    print(f"Objective value: {tt_inner_prod(C, X)}")
    evaled_constraints = tt_eval_constraints(As, X)
    scaled_error = [c / 2 for c in tt_add(evaled_constraints, tt_scale(-1, bias))]
    avg_error = np.sqrt(tt_inner_prod(scaled_error, scaled_error))
    print(f"Avg constraint error: {avg_error}")
    print("Ranks of X: ", tt_ranks(X))
    solution = tt_op_to_matrix(X)
    #print(np.round(tt_op_to_matrix([c.reshape(c.shape[0], 2, 2, c.shape[-1]) for c in tt_rank_reduce(tt_add(I_wtho_lead, one_frame))]), decimals=2))
    #solution = solution[1:, 1:]/ solution[0,  0]
    #nodes_in_cut = [i for i, v in enumerate(solution[0]) if v > 0.01]
    #adj_matrix = 2*(tt_op_to_matrix(G)[1:, 1:] - 0.5)
    #plot_maxcut(adj_matrix, nodes_in_cut, duality_gaps)
