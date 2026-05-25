import sys
import os
import numpy as np

sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
from src.utils import run_experiment


def tt_diag_constraint_op(dim):
    identity = tt_identity(dim)
    basis = tt_diag_op(identity)
    return basis, identity

def tt_obj_matrix(rank, dim):
    graph_tt = tt_rank_reduce(tt_random_graph(dim, rank))
    laplacian_tt = tt_sub(tt_diag(tt_fast_matrix_vec_mul(graph_tt, [np.ones((1, 2, 1)) for _ in range(dim)],  1e-12)), graph_tt)
    return laplacian_tt

def create_problem(dim, rank):
    print(f"Creating Problem for dim={dim}, rank={rank}...")
    scale = np.sqrt(dim)
    obj_tt = tt_obj_matrix(rank, dim)
    L_tt, bias_tt = tt_diag_constraint_op(dim)
    lag_y = tt_diag_op(tt_sub(tt_one_matrix(dim), tt_identity(dim)))
    return tt_reshape(tt_normalise(obj_tt, radius=scale), (4,)), L_tt, tt_reshape(tt_normalise(bias_tt, radius=scale), (4,)), lag_y


if __name__ == "__main__":
    run_experiment(create_problem)