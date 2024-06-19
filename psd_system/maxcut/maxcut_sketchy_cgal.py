# Import packages.
import sys
import os
sys.path.append(os.getcwd() + '/../../')
import time
from src.tt_ops import *
from psd_system.graph_plotting import *
from maxcut import Config
from src.baselines import *


if __name__ == "__main__":
    np.random.seed(Config.seed)
    t0 = time.time()
    G = tt_random_graph(Config.ranks)
    t1 = time.time()
    print(f"Random graph produced in {t1 - t0:.3f}s")
    C = np.round(tt_op_to_matrix(G))
    t2 = time.time()
    constraint_matrices = [np.outer(column, column) for column in np.eye(C.shape[0])]
    bias = np.ones((C.shape[0], 1))
    trace_param = np.sum(bias)
    X, duality_gaps = sketchy_cgal(-C, constraint_matrices, bias, (trace_param, trace_param), R=2, num_iter=150)
    t3 = time.time()
    print(f"Problem solved in {t3 - t2:.3f}s")
    print(f"Objective value: {np.trace(C.T @ X)}")
    chol = robust_cholesky(X, epsilon=1e-3)
    nodes_in_cut = [i for i, v in enumerate(chol.T @ np.random.randn(chol.shape[0], 1)) if v > 0]
    plot_maxcut(C, nodes_in_cut, duality_gaps)