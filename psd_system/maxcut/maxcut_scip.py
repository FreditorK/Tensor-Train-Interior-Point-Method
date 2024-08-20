import sys
import os
sys.path.append(os.getcwd() + '/../../')
import cvxpy as cp
import time
from src.tt_ops import *
from psd_system.graph_plotting import *
from maxcut import Config


if __name__ == "__main__":
    np.random.seed(Config.seed)
    t0 = time.time()
    G = tt_random_graph(Config.ranks)
    t1 = time.time()
    print(f"Random graph produced in {t1 - t0:.3f}s")
    C = np.round(tt_matrix_to_matrix(G))
    X = cp.Variable(C.shape, symmetric=True)
    constraints = [X >> 0]
    constraints += [cp.diag(X) == 1]
    t2 = time.time()
    prob = cp.Problem(cp.Maximize(cp.trace(C @ X)), constraints)
    prob.solve()
    t3 = time.time()
    print(np.round(X.value, decimals=2))
    print(f"Problem solved in {t3 - t2:.3f}s")
    print(f"Objective value: {prob.value}")
    chol = robust_cholesky(X.value, epsilon=1e-3)
    nodes_in_cut = [i for i, v in enumerate(chol.T @ np.random.randn(chol.shape[0], 1)) if v > 0]
    plot_maxcut(C, nodes_in_cut, [])

