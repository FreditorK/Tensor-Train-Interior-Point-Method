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
from src.utils import print_results_summary


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
    obj_val = float(np.trace(C @ X))

    # Lagrange multipliers from first-order conditions, y_i = (C X)_{ii}.
    y = np.einsum("ij,ij->i", C, X)
    Z = np.diag(y) - C
    stationarity_error = float(np.linalg.norm(U @ Z) ** 2)
    duality_gap = float(abs(np.sum(y) - obj_val))
    primal_feasibility_error = float(np.linalg.norm(np.diag(X) - 1.0) ** 2)
    neg_eigs = np.minimum(np.linalg.eigvalsh(Z), 0.0)
    dual_psd_violation = float(np.dot(neg_eigs, neg_eigs))

    return (
        stationarity_error,
        duality_gap,
        primal_feasibility_error,
        dual_psd_violation,
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


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    parser = argparse.ArgumentParser(description="Riemannian MaxCut baseline with optional memory tracking.")
    parser.add_argument("--track_mem", action="store_true", help="Enable memory tracking from a certain point.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
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
    duality_gaps = np.zeros(num_seeds)
    primal_feasibility_errors = np.zeros(num_seeds)
    dual_psd_violations = np.zeros(num_seeds)
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
            C = tt_matrix_to_matrix(tt_obj_matrix(1, config["dim"]))
            t2 = time.time()

            try:
                (
                    stationarity_err,
                    duality_gap,
                    primal_feas_err,
                    dual_psd_violation,
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
        dual_psd_violations[s_i] = dual_psd_violation
        duality_gaps[s_i] = duality_gap
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
    print("\n" + "=" * 80)
    print("IMPORTANT: MANOPT METRIC LABEL MAPPING")
    print("  Feasibility Error      -> Primal feasibility ||diag(X)-1||^2")
    print("  Dual Feasibility Error -> Dual PSD violation of Z (Z ⪰ 0)")
    print("  Duality Gap            -> |1^T y - tr(CX)|")
    print("  Stationarity (printed above) -> ||U Z||^2")
    print("=" * 80 + "\n")

    # Keep compatibility with shared summary utility:
    # feasibility_errors -> primal feasibility, dual_feasibility_errors -> Z PSD violation,
    # complementary_slackness -> duality gap.
    print_results_summary(
        config,
        args,
        runtimes.reshape(1, -1),
        problem_creation_times.reshape(1, -1),
        num_iters.reshape(1, -1),
        primal_feasibility_errors.reshape(1, -1),
        dual_psd_violations.reshape(1, -1),
        duality_gaps.reshape(1, -1),
        ranksX,
        ranksY,
        ranksZ,
        ranksT=None,
        memory=memory.reshape(1, -1) if args.track_mem else None,
    )
