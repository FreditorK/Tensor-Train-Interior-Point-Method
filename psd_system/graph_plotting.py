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
    seed = 33

np.random.seed(Config.seed)

graphs = []
ranks = [[1, 1], [1, 8], [10, 5], [10, 10]]
for i, rank in enumerate(ranks):
    tensor = tt_random_graph(rank)
    ranks[i] = tt_ranks(tensor)[::2]
    matrix = np.round(np.round(tt_op_to_matrix(tensor), decimals=2))
    G = nx.from_numpy_matrix(matrix)
    graphs.append(G)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for i, (rank, G) in enumerate(zip(ranks, graphs)):
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, ax=axes[i], with_labels=True, node_color='lightblue', edge_color='gray', node_size=300, font_size=10, font_color='black')
    axes[i].set_title(f'Graph {rank}')
plt.show()