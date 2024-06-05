# https://epubs.siam.org/doi/epdf/10.1137/19M1305045
import numpy as np


def cgal(obj_matrix, constraint_matrices, bias, trace_param, num_iter=100):
    X = np.zeros_like(obj_matrix)
    res = -bias
    lag_mul_1 = np.zeros((len(constraint_matrices), 1))
    lag_mul_2 = 1
    alpha_0 = 4 * np.sum([np.linalg.norm(A) for A in constraint_matrices]) * trace_param**2
    duality_gaps = []
    for it in range(1, num_iter):
        constraint_term = sum([A.T*(y_i + lag_mul_2*r) for A, y_i, r in zip(constraint_matrices, lag_mul_1.flatten(), res.flatten())])
        sdp_gradient = obj_matrix + constraint_term
        eig, min_eig_val = power_method(2*np.eye(sdp_gradient.shape[0]) - sdp_gradient)
        eta = np.divide(2, it + 1)
        print(np.trace(obj_matrix @ X), np.trace(constraint_term @ X), trace_param, min_eig_val)
        duality_gap = np.trace(obj_matrix @ X) + np.trace(constraint_term @ X) - trace_param*(2 - min_eig_val)
        duality_gaps.append(duality_gap)
        X = (1 -eta)*X + eta*trace_param*np.outer(eig, eig)
        res = np.array([np.trace(A.T @ X) - b for A, b in zip(constraint_matrices, bias.flatten())]).reshape(-1, 1)
        gamma = min(np.divide(alpha_0, np.power(it + 1, 3/2) * (res.T @ res)), 1)
        lag_mul_1 = lag_mul_1 + gamma*res
        lag_mul_2 = np.sqrt(it+1)
    return X, duality_gaps


def power_method(matrix, num_iter=200):
    v = np.random.randn(matrix.shape[0], 1)
    for _ in range(num_iter):
        v = matrix @ v
        v = np.divide(v, np.linalg.norm(v))
    prev_v = v
    v = matrix @ v
    return np.divide(v, np.linalg.norm(v)), (prev_v.T @ v).item()


def approx_min_evec(matrix, num_iter=1000):
    v = np.random.randn(matrix.shape[0], 1)
    v = np.divide(v, np.linalg.norm(v))
    prev_v = np.zeros_like(v)
    rho = 0
    rhos, ws, vs = [], [], []
    num_iter = min(num_iter, matrix.shape[0])
    for i in range(num_iter):
        vs.append(v)
        w = v.T @ matrix @ v
        ws.append(w.item())
        new_v = matrix @ v - w*v - rho*prev_v
        prev_v = v
        rho = np.linalg.norm(new_v)
        rhos.append(rho.item())
        if np.equal(rho, 0):
            break
        v = np.divide(new_v, rho)
    vs.append(v)
    tridiagonal_matrix = np.zeros((num_iter, num_iter))
    np.fill_diagonal(tridiagonal_matrix, ws)
    np.fill_diagonal(tridiagonal_matrix[0:-1, 1:], rhos[:-1])
    np.fill_diagonal(tridiagonal_matrix[1:, 0:-1], rhos[:-1])
    eig_vals, eig_vecs = np.linalg.eig(tridiagonal_matrix)
    min_idx = np.argmin(eig_vals)
    min_eig = eig_vecs[min_idx]
    v = np.sum([u*v for u, v in zip(min_eig, vs)])
    return eig_vals[min_idx], v