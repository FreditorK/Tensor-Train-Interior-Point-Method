import sys
import os
import time

import numpy as np

sys.path.append(os.getcwd() + '/../')
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from src.tt_op import *

@dataclass
class Config:
    seed = 49
    ranks = [3, 3]

np.random.seed(Config.seed)

G = tt_random_graph(Config.ranks)
G = tt_scale(-1, G)
print([g.shape for g in G])
diag_core = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]]).reshape(1, 2, 2, 2, 1)
As = [diag_core]*len(G)
bias = tt_one(len(G))
t0 = time.time()
X = tt_sdp_fw(G, As, bias, trace_param=2**len(bias), bt=False, num_iter=100)
t1 = time.time()
print(f"Problem solved in {t1-t0}s")
print(f"Objective value: {tt_inner_prod(G, X)}")
evaled_constraints = tt_eval_constraints(As, X)
scaled_error = [c / 2 for c in tt_add(evaled_constraints, tt_scale(-1, bias))]
avg_error = np.sqrt(tt_inner_prod(scaled_error, scaled_error))
print(f"Avg error: {avg_error}")
print("Ranks of X: ", tt_ranks(X))

solution = tt_op_to_matrix(X)
chol = np.linalg.cholesky(solution)
nodes_in_cut = [i for i, vec in enumerate(chol.T) if vec @ np.random.randn(*vec.shape).T > 0]
solution = np.clip(100*solution, a_max=1, a_min=0)
matrix = (-np.round(tt_op_to_matrix(G)) + 1)/2
masked_solution = matrix*solution
cut_edges = sum([[(i, j) for j in range(len(matrix)) if masked_solution[i, j] > 0.5] for i in range(len(matrix))], [])
graph = nx.from_numpy_matrix(matrix)
pos = nx.spring_layout(graph)
nx.draw_networkx(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=300, font_size=10, font_color='black')
nx.draw_networkx_nodes(graph, pos, nodelist=nodes_in_cut, alpha=1.0, node_color='r')
plt.show()
