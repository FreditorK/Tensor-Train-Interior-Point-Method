import sys
import os
import time
import argparse
import tracemalloc
import yaml

sys.path.append(os.getcwd() + '/../../')

from src.tt_ops import *
from src.tt_ipm import tt_ipm


def tt_diag_op(dim):
    identity = tt_identity(dim)
    basis = tt_diag(tt_split_bonds(identity))
    return basis

def tt_diag_op_adj(dim):
    return tt_diag_op(dim)


def tt_obj_matrix(rank, dim):
    graph_tt = tt_rank_reduce(tt_random_graph(dim, rank))
    laplacian_tt = tt_sub(tt_diag(tt_fast_matrix_vec_mul(graph_tt, [np.ones((1, 2, 1)) for _ in range(dim)],  1e-12)), graph_tt)
    return tt_normalise(laplacian_tt, radius=1)


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument('--rank', type=int, required=False, help='An integer input', default=0)
    args = parser.parse_args()
    with open(os.getcwd() + '/../../' + args.config, "r") as file:
        config = yaml.safe_load(file)

    problem_creation_times = []
    runtimes = []
    memory = []
    complementary_slackness = []
    feasibility_errors = []
    num_iters = []
    for seed in config["seeds"]:
        print("Seed: ", seed)
        np.random.seed(seed)
        t0 = time.time()
        rank = config["max_rank"] if args.rank == 0 else args.rank
        G_tt = tt_obj_matrix(rank, config["dim"])
        t1 = time.time()
        L_tt = tt_diag_op(config["dim"])
        bias_tt = tt_identity(config["dim"])

        lag_maps = {"y": tt_rank_reduce(tt_diag(tt_split_bonds(tt_sub(tt_one_matrix(config["dim"]), tt_identity(config["dim"])))))}

        lag_maps = {key: tt_reshape(value, (4, 4)) for key, value in lag_maps.items()}
        G_tt = tt_reshape(G_tt, (4,))
        L_tt = tt_reshape(L_tt, (4, 4))
        bias_tt = tt_reshape(bias_tt, (4,))

        if args.track_mem:
            tracemalloc.start()
        t2 = time.time()
        X_tt, Y_tt, T_tt, Z_tt, info = tt_ipm(
            lag_maps,
            G_tt,
            L_tt,
            bias_tt,
            max_iter=config["max_iter"],
            verbose=config["verbose"],
            feasibility_tol=config["feasibility_tol"],
            centrality_tol=config["centrality_tol"],
            op_tol=config["op_tol"],
            aho_direction=False
        )
        t3 = time.time()
        if args.track_mem:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()  # Stop tracking after measuring
            memory.append(peak / 10 ** 6)
        problem_creation_times.append(t2 - t1)
        runtimes.append(t3 - t2)
        complementary_slackness.append(abs(tt_inner_prod(X_tt, Z_tt)))
        primal_res = tt_rank_reduce(tt_sub(tt_fast_matrix_vec_mul(L_tt, tt_reshape(X_tt, (4,))), bias_tt),
                                    rank_weighted_error=True, eps=1e-12)

        feasibility_errors.append(tt_inner_prod(primal_res, primal_res))
        num_iters.append(info["num_iters"])
        print(f"Converged after {num_iters[-1]:.1f} iterations", flush=True)
        print(f"Problem created in {problem_creation_times[-1]:.3f}s", flush=True)
        print(f"Problem solved in {runtimes[-1]:.3f}s", flush=True)
        if args.track_mem:
            print(f"Peak memory avg {memory[-1]:.3f} MB", flush=True)
        print(f"Complementary Slackness: {complementary_slackness[-1]}", flush=True)
        print(f"Total feasibility error: {feasibility_errors[-1]}", flush=True)
    print(f"Converged after avg {np.mean(num_iters):.1f} iterations", flush=True)
    print(f"Problem created in avg {np.mean(problem_creation_times):.3f}s", flush=True)
    print(f"Problem solved in avg {np.mean(runtimes):.3f}s", flush=True)
    if args.track_mem:
        print(f"Peak memory avg {np.mean(memory):.3f} MB", flush=True)
    print(f"Complementary Slackness avg: {np.mean(complementary_slackness)}", flush=True)
    print(f"Total feasibility error avg: {np.mean(feasibility_errors)}", flush=True)