# Import packages.
import sys
import os

sys.path.append(os.getcwd() + '/../../')
from src.baselines import *
from maxcut import *


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    parser = argparse.ArgumentParser(description="Script with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    with open(os.getcwd() + '/../../' + args.config, "r") as file:
        config = yaml.safe_load(file)

    problem_creation_times = []
    runtimes = []
    memory = []
    aux_duality_gap = []
    feasibility_errors = []
    num_iters = []
    for seed in config["seeds"]:
        np.random.seed(seed)
        t0 = time.time()
        trace_param = 2**config["dim"]
        C = trace_param*tt_matrix_to_matrix(tt_obj_matrix(config["max_rank"], config["dim"]))
        t1 = time.time()
        constraint_matrices = [np.outer(column, column) for column in np.eye(C.shape[0])]
        bias = np.ones((C.shape[0], 1))
        if args.track_mem:
            start_mem = memory_usage(max_usage=True)
        t2 = time.time()
        if args.track_mem:
            def wrapper():
                X, duality_gaps, info = sketchy_cgal(-C, constraint_matrices, bias, (trace_param, trace_param),
                                                     feasability_tol=config["feasibility_tol"],
                                                     duality_tol=config["duality_tol"],
                                                     num_iter=1000 * 2 ** config["dim"], R=config["sketch_cgal_rank"],
                                                     verbose=config["verbose"])
                return X, duality_gaps, info

            res = memory_usage(proc=wrapper, max_usage=True, retval=True)
            X, duality_gaps, info = res[1]
            memory.append(res[0] - start_mem)
        else:
            X, duality_gaps, info = sketchy_cgal(-C, constraint_matrices, bias, (trace_param, trace_param),
                                                 feasability_tol=config["feasibility_tol"],
                                                 duality_tol=config["duality_tol"],
                                                 num_iter=1000 * 2 ** config["dim"], R=config["sketch_cgal_rank"],
                                                 verbose=config["verbose"])
        t3 = time.time()
        problem_creation_times.append(t2 - t1)
        runtimes.append(t3 - t2)
        aux_duality_gap.append(duality_gaps[-1])
        feasibility_errors.append(
            np.linalg.norm([np.trace(c.T @ X) - b for c, b in zip(constraint_matrices, bias.flatten())]) ** 2)
        num_iters.append(info["num_iters"])
    print(f"Converged after avg {np.mean(num_iters):.1f} iterations")
    print(f"Problem created in avg {np.mean(problem_creation_times):.3f}s")
    print(f"Problem solved in avg {np.mean(runtimes):.3f}s")
    print(f"Peak memory avg {np.mean(memory):.3f} MB")
    print(f"Aux duality gap avg: {np.mean(aux_duality_gap)}")
    print(f"Total feasibility error avg: {np.mean(feasibility_errors)}")