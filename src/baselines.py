import scipy as scp
import numpy as np
from collections import deque

def cgal(obj_matrix, constraint_matrices, bias, trace_params, R=1, gap_tol=1e-5, abs_tol=(0.1, 1e-3), num_iter=100, verbose=False):
    # https://proceedings.mlr.press/v97/yurtsever19a/yurtsever19a.pdf
    feasability_tol = gap_tol
    X = np.zeros_like(obj_matrix)
    bias_norm = np.linalg.norm(bias)
    res = -bias
    lag_mul_1 = np.zeros((len(constraint_matrices), 1))
    lag_mul_2 = 1
    alpha_0 = 4 * np.sum([np.linalg.norm(A) for A in constraint_matrices]) * trace_params[1] ** 2
    duality_gaps = []
    trace = 0
    ncv = 20
    for it in range(1, num_iter):
        constraint_term = sum(
            [A.T * (y_i + lag_mul_2 * r) for A, y_i, r in zip(constraint_matrices, lag_mul_1.flatten(), res.flatten())])
        sdp_gradient = obj_matrix + constraint_term
        try:
            min_eig_val, eig = scp.sparse.linalg.eigsh(sdp_gradient, k=1, which='SA', ncv=ncv)
        except Exception as e:
            print(e)
            ncv = min(ncv+5, 500)
            continue
        eta = np.divide(2, it + 1)
        current_trace_param = trace_params[0] if min_eig_val > 0 else trace_params[1]
        duality_gap = np.squeeze(np.trace(obj_matrix @ X) + np.trace(constraint_term @ X) - current_trace_param * min_eig_val)
        X = (1 - eta) * X + eta * current_trace_param * np.outer(eig, eig)
        trace = (1-eta)*trace + eta*current_trace_param
        res = np.array([np.trace(A.T @ X) - b for A, b in zip(constraint_matrices, bias.flatten())]).reshape(-1, 1)
        alpha = min(np.divide(alpha_0, np.power(it + 1, 3 / 2) * (res.T @ res)), 1)
        lag_mul_1 = lag_mul_1 + alpha * res
        lag_mul_2 = np.sqrt(it + 1)
        if verbose and it % 1000 == 0:
            duality_gaps.append(duality_gap)
            print(f"---Step {it}---")
            print(f"Duality gap: {np.sum(duality_gap)}")
            print(f"Feasibility error: {np.linalg.norm(res)**2}", flush=True)
        res_norm = np.sqrt(res.T @ res)
        if duality_gap < min((1+abs(np.trace(obj_matrix @ X)))*gap_tol, abs_tol[0]) and  res_norm < min((1+bias_norm)*feasability_tol, abs_tol[1]):
            break
    if verbose:
        print("Converged after {} iterations".format(it))
    return X, duality_gaps, {"num_iters": it}


def sketchy_cgal(obj_matrix, constraint_matrices, bias, trace_params, R=1, gap_tol=1e-5, abs_tol=(0.1, 1e-3), num_iter=100, verbose=False):
    # https://epubs.siam.org/doi/epdf/10.1137/19M1305045
    feasability_tol = gap_tol
    bias_norm = np.linalg.norm(bias)
    Omega, S = nystrom_sketch_init(obj_matrix.shape[0], R)
    res = -bias
    lag_mul_1 = np.zeros((len(constraint_matrices), 1))
    z = np.zeros_like(lag_mul_1)
    lag_mul_2 = 1
    alpha_0 = 4 * np.sum([np.linalg.norm(A) for A in constraint_matrices]) * trace_params[1] ** 2
    duality_gaps = deque(maxlen=10)
    stagnation_window = 100  # You can make this a parameter if you wish
    stagnation_tol = 1e-6   # You can make this a parameter if you wish
    p = 0
    current_trace = 0
    ncv = 20
    best_gap = float('inf')
    no_improve_count = 0
    improve_window = 50

    for it in range(1, num_iter):
        constraint_term = sum(A.T * (y_i + lag_mul_2 * r) for A, y_i, r in zip(constraint_matrices, lag_mul_1.flatten(), res.flatten()))
        sdp_gradient = obj_matrix + constraint_term
        try:
            min_eig_val, eig = scp.sparse.linalg.eigsh(sdp_gradient, k=1, which='SA', ncv=ncv)
        except Exception as e:
            print(e)
            ncv = min(ncv+5, 500)
            continue
        eta = np.divide(2, it + 1)
        current_trace_param = trace_params[0] if min_eig_val > 0 else trace_params[1]
        current_trace = (1- eta)*current_trace + eta*current_trace_param
        p = (1-eta)*p + eta*current_trace_param*(eig.T @ obj_matrix @ eig).item()
        duality_gap = np.squeeze(p + ((lag_mul_1 + lag_mul_2*res).T @ z).item() - current_trace_param * min_eig_val)
        duality_gaps.append(duality_gap)
        z = (1 - eta) * z + eta * current_trace_param * np.array([eig.T @ A.T @ eig for A in constraint_matrices]).reshape(-1, 1)
        res = z - bias
        S = nystrom_sketch_update(S, Omega, eig, eta, current_trace_param)
        alpha = min(np.divide(alpha_0, np.power(it + 1, 3 / 2) * (res.T @ res)), 1)
        lag_mul_1 = lag_mul_1 + alpha * res
        lag_mul_2 = np.sqrt(it + 1)
        if verbose and it % 1000 == 0:
            print(f"---Step {it}---")
            print(f"Est. Duality gap: {duality_gap}")
            print(f"Est. Feasibility error: {np.sum(res.T @ res)}", flush=True)
        res_norm = np.sqrt(res.T @ res)
        if (duality_gap < min((1+abs(p))*gap_tol, abs_tol[0]) and  res_norm < min((1+bias_norm)*feasability_tol, abs_tol[1])):
            break
        elif len(duality_gaps) == stagnation_window:
            mean_abs_change = np.mean(np.abs(np.diff(duality_gaps)))
            std_gap = np.std(duality_gaps)
            if mean_abs_change < stagnation_tol and std_gap < stagnation_tol:
                if verbose:
                    print(f"Breaking due to stagnation (mean abs change & std dev) at iteration {it}")
                break
            if duality_gap < best_gap:
                best_gap = duality_gap
                no_improve_count = max(0, no_improve_count - 0.5)
            else:
                no_improve_count += 1
            if no_improve_count >= improve_window:
                if verbose:
                    print(f"Breaking due to no improvement at iteration {it}")
                break
    U, Lambda = nystrom_sketch_reconstruct(S, Omega)
    U = U[:, :R]
    Lambda = Lambda + (current_trace - np.trace(Lambda)) * np.eye(R) / R
    if verbose:
        print("Converged after {} iterations".format(it))
        print(f"Est. Duality gap: {duality_gap}")
        print(f"Est. Feasibility error: {np.sum(res.T @ res)}")
    X = U @ Lambda @ U.T
    min_eig_val, eig = scp.sparse.linalg.eigsh(sdp_gradient, k=1, which='SA', ncv=ncv)
    current_trace_param = trace_params[0] if min_eig_val > 0 else trace_params[1]
    duality_gap = np.squeeze(np.trace(obj_matrix @ X) + np.trace(constraint_term @ X) - current_trace_param * min_eig_val)
    duality_gaps.append(duality_gap)
    return X, duality_gaps, {"num_iters": it}


def nystrom_sketch_init(n, R):
    Omega = np.random.randn(n, R)
    S = np.zeros((n, R))
    return Omega, S


def nystrom_sketch_update(S, Omega, eig, eta, trace_param):
    S = (1 - eta) * S + eta * trace_param * np.outer(eig, eig.T @ Omega)
    return S


def nystrom_sketch_reconstruct(Y, Omega, tol=2.2e-12):
    sigma = tol * np.sqrt(Y.shape[0]) * np.linalg.norm(Y)
    Y = Y + sigma * Omega
    C = np.linalg.cholesky(Omega.T @ Y)
    B = Y @ np.linalg.inv(C.T)
    U, d, _ = np.linalg.svd(B)
    Sigma = np.diag(d)
    Lambda = np.maximum(0, Sigma**2 - sigma * np.eye(Sigma.shape[0]))
    U = U[:, :len(Lambda)]
    return U, Lambda
