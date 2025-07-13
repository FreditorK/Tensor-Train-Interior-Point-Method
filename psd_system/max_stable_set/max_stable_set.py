import copy
import sys
import os

sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
from src.utils import run_experiment


def tt_G_entrywise_mask_op(G):
    vec_g_copy = tt_split_bonds(copy.deepcopy(G))
    basis = []
    for g_core in vec_g_copy:
        core = np.zeros((g_core.shape[0], 2, 2, g_core.shape[-1]))
        core[:, 0, 0] = g_core[:, 0]
        core[:, 1, 1] = g_core[:, 1]
        basis.append(core)
    return tt_rank_reduce(tt_reshape(basis, (4, 4)))

def tt_tr_constraint(dim):
    op =[]
    for i, c in enumerate(tt_split_bonds([np.eye(2).reshape(1, 2, 2, 1) for _ in range(dim)])):
        core = np.zeros((c.shape[0], 2, 2, c.shape[-1]))
        core[:, 0] = c
        op.append(core)
    return tt_rank_reduce(tt_reshape(op, (4, 4))), [E(0, 0) for _ in range(dim)]

def tt_obj_matrix(dim):
    return tt_one_matrix(dim)


def create_problem(dim, rank):
    print("Creating Problem...")
    scale = max(2**(dim-7), 1)
    G = tt_rank_reduce(tt_random_graph(dim, rank))
    obj_tt = tt_obj_matrix(dim)
    L_tt, bias_tt = tt_tr_constraint(dim)
    L_tt = tt_rank_reduce(tt_add(L_tt, tt_G_entrywise_mask_op(G)))
    lag_y = tt_rank_reduce(tt_diag_op(tt_sub(tt_one_matrix(dim), tt_add(G, bias_tt))))
    return tt_reshape(tt_normalise(obj_tt, radius=scale), (4,)), L_tt, tt_reshape(tt_normalise(bias_tt, radius=scale), (4,)), lag_y

if __name__ == "__main__":
    run_experiment(create_problem)