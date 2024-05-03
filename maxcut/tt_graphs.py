import os
import sys

sys.path.append(os.getcwd() + '/../')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from src.tt_op import *


def show_graph_with_labels(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw_networkx(gr, with_labels=True)
    plt.show()

"""
matrix = np.random.randint(size=(8, 8), low=0, high=2)
matrix = (matrix + matrix.T) % 2
print(matrix)
tt_tensor = tt_svd(matrix.reshape(2, 2, 2, 2, 2, 2))
print([c.shape for c in tt_tensor])
show_graph_with_labels(matrix)
"""
chordal_matrix = np.array([
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 1, 0, 1],
    [1, 1, 0, 0, 1, 1, 1, 0]
])
print(chordal_matrix)
blocks = [
    np.pad(chordal_matrix[:4, :4], ((0, 4), (0, 4))),
    np.pad(chordal_matrix[:4, 4:], ((0, 4), (4, 0))),
    np.pad(chordal_matrix[4:, :4], ((4, 0), (0, 4))),
    np.pad(chordal_matrix[4:, 4:], ((4, 0), (4, 0)))
]

tt_blocks = [[np.zeros((1, 2, 1)) for _ in range(6)]] + [
    tt_svd(b.reshape(2, 2, 2, 2, 2, 2)) for b in blocks[1:]
]

tensor_matrix, index_length = tt_tensor_matrix(tt_blocks)
print([c.shape for c in tensor_matrix])
show_graph_with_labels(chordal_matrix)

