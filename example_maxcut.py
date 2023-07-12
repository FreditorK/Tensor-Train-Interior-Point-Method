import networkx as nx
import matplotlib.pyplot as plt
from functools import reduce
from time import time
from utils import *
from optimiser import ILPSolver, AnswerSetSolver

vocab_size = 3  # i.e. graph with 2^3 nodes
atoms = np.array(generate_atoms(vocab_size))

"""
def coin_symmetry(tt_train):
    return tt_inner_prod(weights, tt_train)
"""
const_space = ConstraintSpace(vocab_size)

graph_edges = [(0, 1), (1, 7), (7, 2), (7, 6), (2, 3), (2, 4), (3, 4), (3, 5), (5, 6)]

graph = tt_svd(-2*graph_to_tensor(3, graph_edges)+1)
bool_graph = tt_bool_op_inv(graph)


def maxcut_objective(tt_train):
    return tt_inner_prod(bool_graph, tt_kronecker_prod(tt_train, tt_train))


opt = ILPSolver(const_space, vocab_size, objective=maxcut_objective)
t_1 = time()
hypothesis = opt.find_feasible_hypothesis()
t_2 = time()
set_partition = tt_to_tensor(tt_bool_op(hypothesis)).flatten()
cut_edges = [e for e in graph_edges if set_partition[e[0]]*set_partition[e[1]] < 0]
print("The cut edges are: ", cut_edges)
G = nx.Graph()
G.add_edges_from(graph_edges)
pos = nx.spring_layout(G)
edge_colors = ['red' if edge in cut_edges else 'black' for edge in G.edges()]
nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color=edge_colors, width=2)
plt.show()
