# Import packages.
import argparse
import os
import sys
import time

import numpy as np
import yaml
from memory_profiler import memory_usage

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from maxcut import tt_matrix_to_matrix, tt_obj_matrix
from src.utils import format_ranks_with_std


def _default_rank(n):
    return max(2, int(np.ceil(np.sqrt(2 * n))))


def _solve_with_pymanopt(C, config):
    import pymanopt
    from pymanopt import Problem
    from pymanopt.manifolds import Oblique
    from pymanopt.optimizers import ConjugateGradient
    from pymanopt.optimizers import TrustRegions

    C = 0.5 * (C + C.T)
    n = C.shape[0]
    rank = min(n, max(2, int(config.get("riem_rank", _default_rank(n)))))
    manifold = Oblique(rank, n)

    @pymanopt.function.numpy(manifold)
    def cost(U):
        return -np.trace(U @ C @ U.T)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(U):
        return -2.0 * U @ C

    @pymanopt.function.numpy(manifold)
    def euclidean_hessian(U, H):
        return -2.0 * H @ C

    max_iterations = int(config.get("riem_max_iter", 500))
    min_grad_norm = float(config.get("riem_min_grad_norm", config.get("gap_tol", 1e-6)))
    problem = Problem(
        manifold,
        cost,
        euclidean_gradient=euclidean_gradient,
        euclidean_hessian=euclidean_hessian,
    )
    opt_kwargs = {
        "min_gradient_norm": min_grad_norm,
        "max_iterations": max_iterations,
        "verbosity": 2 if config.get("verbose", False) else 0,
    }
    try:
        result = TrustRegions(**opt_kwargs).run(problem)
    except NotImplementedError as err:
        if "No autodiff support available for the NumPy backend" not in str(err):
            raise
        if config.get("verbose", False):
            print("TrustRegions NumPy backend fallback: switching to ConjugateGradient.")
        result = ConjugateGradient(**opt_kwargs).run(problem)

    U = result.point
    X = U.T @ U

    # Lagrange multipliers from first-order conditions, y_i = (C X)_{ii}.
    y = np.einsum("ij,ij->i", C, X)
    Z = np.diag(y) - C
    stationarity_error = float(np.linalg.norm(U @ Z) ** 2)
    primal_feasibility_error = float(np.linalg.norm(np.diag(X) - 1.0) ** 2)

    return (
        stationarity_error,
        primal_feasibility_error,
        result.iterations,
        rank,
    )


def _solve_with_optional_mem(C, config, track_mem):
    if not track_mem:
        return (*_solve_with_pymanopt(C, config), 0.0)
    start_mem = memory_usage(max_usage=True, include_children=True)
    max_mem, result = memory_usage(
        proc=lambda: _solve_with_pymanopt(C, config),
        max_usage=True,
        retval=True,
        include_children=True,
    )
    return (*result, max_mem - start_mem)


def _print_manopt_results_summary(
    config,
    args,
    runtimes,
    problem_creation_times,
    num_iters,
    primal_feasibility_errors,
    stationarity_errors,
    ranksX,
    ranksY,
    ranksZ,
    memory=None,
):
    print("\n" + "=" * 80)
    print(f"{'FINAL RESULTS SUMMARY':^80}")
    print("=" * 80)
    print(f"Values are reported as Mean ± Standard Deviation over all seeds {config['seeds']}.\n")
    print("  MANOPT note: Dual Feasibility Error and Duality Gap are reported as '--'.\n")

    print(f"  {'Metric':<28} | {'Value (Mean ± Std)':>25}")
    print(f"  {'-' * 28} | {'-' * 25}")
    print(f"  {'Solution Time (s)':<28} | {f'{np.mean(runtimes):.3f} ± {np.std(runtimes):.3f}':>25}")
    print(
        f"  {'Problem Creation (s)':<28} | "
        f"{f'{np.mean(problem_creation_times):.3f} ± {np.std(problem_creation_times):.3f}':>25}"
    )
    print(f"  {'Iterations':<28} | {f'{np.mean(num_iters):.1f} ± {np.std(num_iters):.1f}':>25}")
    print(
        f"  {'Feasibility Error':<28} | "
        f"{f'{np.mean(primal_feasibility_errors):.2e} ± {np.std(primal_feasibility_errors):.2e}':>25}"
    )
    print(f"  {'Dual Feasibility Error':<28} | {'--':>25}")
    print(f"  {'Duality Gap':<28} | {'--':>25}")
    print(
        f"  {'Stationarity ||U Z||^2':<28} | "
        f"{f'{np.mean(stationarity_errors):.2e} ± {np.std(stationarity_errors):.2e}':>25}"
    )

    if args.track_mem and memory is not None:
        print(f"  {'Peak Memory (MB)':<28} | {f'{np.mean(memory):.3f} ± {np.std(memory):.3f}':>25}")

    print(f"  {'-' * 28} | {'-' * 25}")
    print(f"  {'Rank Statistics':<55}")
    print(f"  {'  Ranks X':<26}: {format_ranks_with_std(np.mean(ranksX[0], axis=0), np.std(ranksX[0], axis=0))}")
    print(f"  {'  Ranks Y':<26}: {format_ranks_with_std(np.mean(ranksY[0], axis=0), np.std(ranksY[0], axis=0))}")
    print(f"  {'  Ranks Z':<26}: {format_ranks_with_std(np.mean(ranksZ[0], axis=0), np.std(ranksZ[0], axis=0))}")
    print("")
    print("=" * 80)


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    parser = argparse.ArgumentParser(description="Riemannian MaxCut baseline with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--rank", type=int, default=1, help="TT-rank used for graph generation")
    args = parser.parse_args()

    try:
        import pymanopt  # noqa: F401
    except ImportError as exc:
        raise ImportError("pymanopt is required for this script. Install with: pip install pymanopt") from exc

    with open(os.path.join(ROOT, args.config), "r") as file:
        config = yaml.safe_load(file)

    num_seeds = len(config["seeds"])
    problem_creation_times = np.zeros(num_seeds)
    runtimes = np.zeros(num_seeds)
    memory = np.zeros(num_seeds)
    primal_feasibility_errors = np.zeros(num_seeds)
    stationarity_errors = np.zeros(num_seeds)
    num_failed_seeds = 0
    num_iters = np.zeros(num_seeds)
    rank_values = []

    for s_i, seed in enumerate(config["seeds"]):
        for attempt in range(3):
            current_seed = seed if attempt == 0 else np.random.randint(0, 10000)
            if attempt > 0:
                print(f"Trying with new random seed: {current_seed}")

            np.random.seed(current_seed)
            t1 = time.time()
            C = tt_matrix_to_matrix(tt_obj_matrix(args.rank, config["dim"]))
            t2 = time.time()

            try:
                (
                    stationarity_err,
                    primal_feas_err,
                    iters,
                    rank,
                    mem_delta,
                ) = _solve_with_optional_mem(C, config, args.track_mem)
                memory[s_i] = mem_delta
                break
            except Exception as err:
                print(err)
                if attempt == 0:
                    print(f"Failed to solve problem with config seed {seed}, trying a new random seed...")
                else:
                    print(f"Failed to solve problem with new random seed {current_seed}")
                    num_failed_seeds += 1
        else:
            continue

        t3 = time.time()
        problem_creation_times[s_i] = t2 - t1
        runtimes[s_i] = t3 - t2
        primal_feasibility_errors[s_i] = primal_feas_err
        stationarity_errors[s_i] = stationarity_err
        num_iters[s_i] = iters
        rank_values.append(rank)

    ranksX = np.zeros((1, num_seeds, 1))
    ranksY = np.zeros((1, num_seeds, 1))
    ranksZ = np.zeros((1, num_seeds, 1))

    print(f"Number of failed seeds: {num_failed_seeds}")
    if rank_values:
        rank_values = np.array(rank_values)
        print(f"Riemannian rank used (mean ± std): {np.mean(rank_values):.1f} ± {np.std(rank_values):.1f}")
    print(
        "Stationarity ||U Z||^2 (mean ± std): "
        f"{np.mean(stationarity_errors):.2e} ± {np.std(stationarity_errors):.2e}"
    )

    _print_manopt_results_summary(
        config=config,
        args=args,
        runtimes=runtimes,
        problem_creation_times=problem_creation_times,
        num_iters=num_iters,
        primal_feasibility_errors=primal_feasibility_errors,
        stationarity_errors=stationarity_errors,
        ranksX=ranksX,
        ranksY=ranksY,
        ranksZ=ranksZ,
        memory=memory if args.track_mem else None,
    )
