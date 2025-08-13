from mmap import ACCESS_COPY
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

def tt_obj_matrix_and_ineq_mask(rank, dim):
    sim_graph_tt = tt_rank_reduce(tt_random_graph(dim, rank), 1e-12)
    disim_graph_tt = tt_rank_reduce(tt_random_graph(dim, rank), 1e-12)
    cap_graph_tt = tt_rank_reduce(tt_fast_hadamard(sim_graph_tt, disim_graph_tt, 1e-12), 1e-12)
    new_sim_graph_tt = tt_sub(sim_graph_tt, cap_graph_tt)
    # Making sure we have two disjoint graphs
    while tt_norm(cap_graph_tt) < 1e-12 or tt_norm(new_sim_graph_tt) < 1e-12:
        disim_graph_tt = tt_rank_reduce(tt_random_graph(dim, rank))
        cap_graph_tt = tt_rank_reduce(tt_fast_hadamard(sim_graph_tt, disim_graph_tt, 1e-12), 1e-12)
        new_sim_graph_tt = tt_sub(sim_graph_tt, cap_graph_tt)
    sim_graph_tt = new_sim_graph_tt
    actual_graph_tt = tt_rank_reduce(tt_add(sim_graph_tt, disim_graph_tt), 1e-12)
    disim_laplacian_tt = tt_sub(tt_diag(tt_fast_matrix_vec_mul(disim_graph_tt, [np.ones((1, 2, 1)) for _ in range(dim)],  1e-12)), disim_graph_tt)
    obj_tt = tt_rank_reduce(tt_add(sim_graph_tt, disim_laplacian_tt), 1e-12)
    print("Actual graph TT-rank:", tt_ranks(actual_graph_tt))
    print("Obj TT-rank:", tt_ranks(obj_tt))
    return obj_tt, actual_graph_tt

def create_problem(dim, rank):
    print(f"Creating Problem for dim={dim}, rank={rank}...")
    scale = max(2**(dim-7), 1)
    obj_tt, ineq_mask = tt_obj_matrix_and_ineq_mask(rank, dim)
    L_tt, bias_tt = tt_diag_constraint_op(dim)
    lag_y = tt_sub(tt_one_matrix(dim), tt_identity(dim))
    lag_t = tt_sub(tt_one_matrix(dim), ineq_mask)
    lag_maps = {
        "y": tt_diag_op(lag_y),
        "t": tt_diag_op(lag_t)
    }
    return tt_reshape(tt_normalise(obj_tt, radius=scale), (4,)), L_tt, tt_reshape(tt_normalise(bias_tt, radius=scale), (4,)), ineq_mask, lag_maps


if __name__ == "__main__":
    run_experiment(create_problem)