import copy
import sys
import os

from numba.np.arraymath import np_quantile

sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from src.tt_ops import _tt_core_collapse, tt_random_gaussian
from psd_system.graph_plotting import *
from src.tt_ipm import tt_ipm, _tt_get_block
import time


@dataclass
class Config:
    seed = 3
    ranks = [3]


if __name__ == "__main__":
    print("Creating Problem...")
    """
        [Q   P  0 ]
    X = [P^T 1  0 ]
        [0   0  I ]
    """
    q_op_prefix = np.zeros((1, 4, 2, 2, 1))
    q_op_prefix[0, 0, 0, 0, 0] = 1
    np.random.seed(Config.seed)
    #graph_A = tt_random_graph(Config.ranks)
    #G_A = tt_scale(0.5, tt_add(graph_A, tt_one_matrix(len(Config.ranks) + 1)))
    #G_A = tt_rank_reduce(G_A)

    #print(np.round(tt_matrix_to_matrix(G_A), decimals=2))

    #graph_B = tt_random_graph(Config.ranks)
    #G_B = tt_scale(0.5, tt_add(graph_B, tt_one_matrix(len(Config.ranks) + 1)))
    #G_B = tt_rank_reduce(G_B)
    #print(np.round(tt_matrix_to_matrix(G_B), decimals=2))

    test_graph = tt_random_gaussian([3], shape=(2, 2)) #tt_random_graph([3])
    #test_graph = tt_scale(0.5, tt_add(test_graph, tt_one_matrix(len(Config.ranks) + 1)))
    G = tt_rank_reduce(test_graph) + [np.ones((1, 2, 2, 1))]
    np.set_printoptions(linewidth=600, threshold=np.inf, precision=4, suppress=True)
    print(np.round(tt_matrix_to_matrix(G), decimals=2))

    n = 1 # 2^1
    n_sq = 2
    partial_tr_op = [q_op_prefix] + tt_partial_trace_op(n, n_sq) # II
    partial_tr_J_op = [q_op_prefix] + tt_partial_J_trace_op(n, n_sq) # III
    diag_block_sum_op = [q_op_prefix] + tt_diag_block_sum_linear_op(n_sq, n) # IV
    #print(np.round(tt_matrix_to_matrix(tt_mat(tt_linear_op(diag_block_sum_op, G))), decimals=2))
    """ V
    [Q_11 Q_12 Q_13 Q_14 | P_11 |   0    0    0]
    [Q_21 Q_22 Q_23 Q_24 | P_21 |   0    0    0]
    [Q_31 Q_32 Q_33 Q_34 | P_12 |   0    0    0]
    [Q_41 Q_42 Q_43 Q_44 | P_22 |   0    0    0]
    [------------------------------------------]
    [P_11 P_21 P_12 P_22 |    1 |   0    0    0]
    [------------------------------------------]
    [   0    0    0    0 |    0 |   1    0    0]
    [   0    0    0    0 |    0 |   0    1    0]
    [   0    0    0    0 |    0 |   0    0    1]
    
    
    
    [   1    0    0    0 | -1/2 |   0    0    0]
    [   0    1    0    0 | -1/2 |   0    0    0]
    [   0    0    1    0 | -1/2 |   0    0    0]
    [   0    0    0    1 | -1/2 |   0    0    0]
    [------------------------------------------]
    [-1/2 -1/2 -1/2 -1/2 |    0 |   0    0    0]
    [------------------------------------------]
    [   0    0    0    0 |    0 |   0    0    0]
    [   0    0    0    0 |    0 |   0    0    0]
    [   0    0    0    0 |    0 |   0    0    0]
    """


    def _core_custom(c, i):
        mask = np.zeros((c.shape[0], 4, *c.shape[1:]))
        if i == 0:
            mask[:, 0] = 0
            mask[:, 1] = c
            mask[:, 2] = 0
            mask[:, 3] = 0
        if i == 1:
            mask[:, 0, 0, 0] = c[:, 0, 0]
            mask[:, 1, 0, 1] = c[:, 0, 1]
            mask[:, 2, 1, 0] = c[:, 1, 0]
            mask[:, 2, 1, 1] = c[:, 1, 1]
        if i == 2:
            mask[:, 0, 0, 0] = c[:, 0, 0]
            mask[:, 1, 0, 1] = c[:, 0, 1]
            mask[:, 2, 1, 0] = c[:, 1, 0]
            mask[:, 2, 1, 1] = c[:, 1, 1]
        return mask


    def tt_custom_op(G, dim):
        v_matrix_1 = [np.array([[1, 0], [0, 0]]).reshape(1, 2, 2, 1)] + tt_identity(n_sq)
        v_matrix_2 = [np.array([[0.0, -0.5], [0.0, 0.0]]).reshape(1, 2, 2, 1)] + [
            np.array([[1.0, 0.0], [1.0, 0.0]]).reshape(1, 2, 2, 1) for _ in range(n_sq)]
        #v_matrix_2 = tt_add(v_matrix_2, tt_transpose(v_matrix_2))
        matrix_tt = tt_rank_reduce(tt_add(v_matrix_1, v_matrix_2))
        print(np.round(tt_matrix_to_matrix(tt_hadamard(matrix_tt, G)), decimals=2))
        return [_core_custom(c, i) for i, c in enumerate(matrix_tt)]

    custom_op = tt_custom_op(G, n_sq)

    print(np.round(tt_matrix_to_matrix(tt_mat(tt_linear_op(custom_op, G))), decimals=2))

