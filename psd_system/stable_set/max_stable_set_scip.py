import sys
import os
import time
import cvxpy as cp

sys.path.append(os.getcwd() + '/../../')

from dataclasses import dataclass
from src.tt_ops import *
from psd_system.graph_plotting import *
from psd_system.stable_set.max_stable_set import Config


if __name__ == "__main__":
    np.random.seed(Config.seed)
    t0 = time.time()
    G = tt_random_graph(Config.ranks)
    adj_matrix = tt_matrix_to_matrix(G)
    adj_matrix = np.round(0.5*(adj_matrix + 1), decimals=1)
    print(adj_matrix)

    t1 = time.time()
    print(f"Random graph produced in {t1 - t0:.3f}s")
    X = cp.Variable(adj_matrix.shape, symmetric=True)
    constraints = [X >> 0]
    constraints += [cp.multiply(adj_matrix, X) == 0]
    constraints += [cp.trace(X) == 1]
    J = np.ones_like(adj_matrix)
    t2 = time.time()
    prob = cp.Problem(cp.Maximize(cp.trace(J @ X)), constraints)
    prob.solve()
    t3 = time.time()
    print(np.round(X.value, decimals=2))
    print(f"Problem solved in {t3 - t2:.3f}s")
    print(f"Objective value: {prob.value}")
