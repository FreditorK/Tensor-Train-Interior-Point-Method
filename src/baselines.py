# https://epubs.siam.org/doi/epdf/10.1137/19M1305045
import numpy as np
import scipy as scp
from src.ops import *


def cgal(obj_matrix, constraint_matrices, bias, trace_params, duality_tol=1e-2, feasability_tol=1e-5, num_iter=100, verbose=False):
    X = np.zeros_like(obj_matrix)
    res = -bias
    lag_mul_1 = np.zeros((len(constraint_matrices), 1))
    lag_mul_2 = 1
    alpha_0 = 4 * np.sum([np.linalg.norm(A) for A in constraint_matrices]) * trace_params[1] ** 2
    duality_gaps = []
    trace = 0
    for it in range(1, num_iter):
        constraint_term = sum(
            [A.T * (y_i + lag_mul_2 * r) for A, y_i, r in zip(constraint_matrices, lag_mul_1.flatten(), res.flatten())])
        sdp_gradient = obj_matrix + constraint_term
        norm = np.linalg.norm(sdp_gradient)
        sdp_gradient = sdp_gradient / norm
        min_eig_val, eig = scp.sparse.linalg.eigsh(2 * np.eye(sdp_gradient.shape[0]) - sdp_gradient, k=1, which='LM')
        min_eig_val = (2 - min_eig_val)*norm
        eta = np.divide(2, it + 1)
        current_trace_param = trace_params[0] if min_eig_val > 0 else trace_params[1]
        duality_gap = np.trace(obj_matrix @ X) + np.trace(constraint_term @ X) - current_trace_param * min_eig_val
        duality_gaps.append(duality_gap)
        X = (1 - eta) * X + eta * current_trace_param * np.outer(eig, eig)
        trace = (1-eta)*trace + eta*current_trace_param
        res = np.array([np.trace(A.T @ X) - b for A, b in zip(constraint_matrices, bias.flatten())]).reshape(-1, 1)
        alpha = min(np.divide(alpha_0, np.power(it + 1, 3 / 2) * (res.T @ res)), 1)
        lag_mul_1 = lag_mul_1 + alpha * res
        lag_mul_2 = np.sqrt(it + 1)
        if verbose:
            print(f"---Step {it}---")
            print(f"Duality gap: {np.sum(duality_gap)}")
            print(f"Feasibility error: {np.linalg.norm(res)**2}")
        if duality_gap < duality_tol and np.linalg.norm(res)**2 < feasability_tol:
            break

    print("Converged after {} iterations".format(it))
    return X, duality_gaps


def sketchy_cgal(obj_matrix, constraint_matrices, bias, trace_params, R=1, duality_tol=1e-2, feasability_tol=1e-5, num_iter=100, verbose=False):
    Omega, S = nystrom_sketch_init(obj_matrix.shape[0], R)
    res = -bias
    lag_mul_1 = np.zeros((len(constraint_matrices), 1))
    z = np.zeros_like(lag_mul_1)
    lag_mul_2 = 1
    alpha_0 = 4 * np.sum([np.linalg.norm(A) for A in constraint_matrices]) * trace_params[1] ** 2
    duality_gaps = []
    p = 0
    current_trace = 0
    for it in range(1, num_iter):
        constraint_term = sum(
            [A.T * (y_i + lag_mul_2 * r) for A, y_i, r in zip(constraint_matrices, lag_mul_1.flatten(), res.flatten())])
        sdp_gradient = obj_matrix + constraint_term
        norm = np.linalg.norm(sdp_gradient)
        sdp_gradient = sdp_gradient / norm
        min_eig_val, eig = scp.sparse.linalg.eigsh(2 * np.eye(sdp_gradient.shape[0]) - sdp_gradient, k=1, which='LM')
        min_eig_val = (2 - min_eig_val) * norm
        eta = np.divide(2, it + 1)
        current_trace_param = trace_params[0] if min_eig_val > 0 else trace_params[1]
        current_trace = (1- eta)*current_trace + eta*current_trace_param
        p = (1-eta)*p + eta*current_trace_param*(eig.T @ (obj_matrix @ eig)).item()
        duality_gap = p + ((lag_mul_1 + lag_mul_2*res).T @ z).item() - current_trace_param * min_eig_val
        duality_gaps.append(duality_gap)
        z = (1 - eta) * z + eta * current_trace_param * np.array([eig.T @ A.T @ eig for A in constraint_matrices]).reshape(-1, 1)
        res = z - bias
        S = nystrom_sketch_update(S, Omega, eig, eta, current_trace_param)
        alpha = min(np.divide(alpha_0, np.power(it + 1, 3 / 2) * (res.T @ res)), 1)
        lag_mul_1 = lag_mul_1 + alpha * res
        lag_mul_2 = np.sqrt(it + 1)
        if verbose:
            print(f"---Step {it}---")
            print(f"Duality gap: {duality_gap}")
            print(f"Feasibility error: {np.sum(res.T @ res)}")
        if duality_gap < duality_tol and res.T @ res < feasability_tol:
            break
    U, Lambda = nystrom_sketch_reconstruct(S, Omega)
    U = U[:, :R]
    Lambda = Lambda + (current_trace - np.trace(Lambda)) * np.eye(R) / R
    print("Converged after {} iterations".format(it))
    X = U @ Lambda @ U.T
    min_eig_val, eig = scp.sparse.linalg.eigsh(2 * np.eye(sdp_gradient.shape[0]) - sdp_gradient, k=1, which='LM')
    min_eig_val = (2 - min_eig_val) * norm
    current_trace_param = trace_params[0] if min_eig_val > 0 else trace_params[1]
    duality_gap = np.trace(obj_matrix @ X) + np.trace(constraint_term @ X) - current_trace_param * min_eig_val
    duality_gaps.append(duality_gap)
    return X, duality_gaps


def power_method(matrix, num_iter=200):
    v = np.random.randn(matrix.shape[0], 1)
    for _ in range(num_iter):
        v = matrix @ v
        v = np.divide(v, np.linalg.norm(v))
    prev_v = v
    v = matrix @ v
    return np.divide(v, np.linalg.norm(v)), (prev_v.T @ v).item()


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
