# Import packages.
from telnetlib import X3PAD
import numpy as np
import argparse
import os
import sys
import time
import yaml
from memory_profiler import memory_usage

sys.path.append(os.getcwd() + '/../../')
from maxcut import *
from src.baselines import *
from src.utils import print_results_summary

if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    with open(os.getcwd() + '/../../' + args.config, "r") as file:
        config = yaml.safe_load(file)

    num_seeds = len(config["seeds"])
    problem_creation_times = np.zeros(num_seeds)
    runtimes = np.zeros(num_seeds)
    memory = np.zeros(num_seeds)
    aux_duality_gap = np.zeros(num_seeds)
    feasibility_errors = np.zeros(num_seeds)
    num_iters = np.zeros(num_seeds)
    num_failed_seeds = 0

    for s_i, seed in enumerate(config["seeds"]):
        tried_new_seed = False
        for attempt in range(3):  # At most two tries: original and one new random seed
            if attempt == 0:
                current_seed = seed
            else:
                current_seed = np.random.randint(0, 10000)
                print(f"Trying with new random seed: {current_seed}")
            np.random.seed(current_seed)
            t1 = time.time()
            trace_param = 2 ** config["dim"]
            C = tt_matrix_to_matrix(tt_obj_matrix(1, config["dim"]))
            C *= trace_param / np.linalg.norm(C)
            t2 = time.time()
            constraint_matrices = [np.outer(column, column) for column in np.eye(C.shape[0])]
            bias = np.ones((C.shape[0], 1))
            sketch_size = 2*int(np.ceil(np.sqrt(2 * (2 ** config["dim"] + 1))))
            try:
                if args.track_mem:
                    start_mem = memory_usage(max_usage=True, include_children=True)
                    def wrapper():
                        X, duality_gaps, info = sketchy_cgal(-C, constraint_matrices, bias, (trace_param, trace_param),
                                                             gap_tol=0.1,
                                                             num_iter=1000 * 2 ** config["dim"],
                                                             R=sketch_size,
                                                             verbose=config["verbose"])
                        return X, duality_gaps, info
                    res = memory_usage(proc=wrapper, max_usage=True, retval=True, include_children=True)
                    X, duality_gaps, info = res[1]
                    memory[s_i] = res[0] - start_mem
                else:
                    X, duality_gaps, info = sketchy_cgal(-C, constraint_matrices, bias, (trace_param, trace_param),
                                                         gap_tol=0.1,
                                                         num_iter=1000 * 2 ** config["dim"],
                                                         R=sketch_size,
                                                         verbose=config["verbose"])
                # If we get here, break out of the attempt loop (success)
                break
            except Exception as e:
                print(e)
                if attempt == 0:
                    print(f"Failed to solve problem with config seed {seed}, trying a new random seed...")
                else:
                    print(f"Failed to solve problem with new random seed {current_seed}")
                    num_failed_seeds += 1
        else:
            # Only runs if all attempts failed
            continue
        t3 = time.time()
        problem_creation_times[s_i] = t2 - t1
        runtimes[s_i] = t3 - t2
        aux_duality_gap[s_i] = duality_gaps[-1]
        feasibility_errors[s_i] = np.linalg.norm([np.trace(c.T @ X) - b for c, b in zip(constraint_matrices, bias.flatten())]) ** 2
        num_iters[s_i] = info["num_iters"]

    # Prepare dummy arrays for missing metrics to match the signature
    ranksX = np.zeros((1, num_seeds, 1))
    ranksY = np.zeros((1, num_seeds, 1))
    ranksZ = np.zeros((1, num_seeds, 1))

    print(f"Number of failed seeds: {num_failed_seeds}")
    print(f"Sketching Size: {sketch_size}")
    print_results_summary(
        config, args,
        runtimes.reshape(1, -1), problem_creation_times.reshape(1, -1), num_iters.reshape(1, -1),
        feasibility_errors.reshape(1, -1), np.zeros((1, num_seeds)), aux_duality_gap.reshape(1, -1),
        ranksX, ranksY, ranksZ,
        ranksT=None,
        memory=memory.reshape(1, -1) if args.track_mem else None
    )