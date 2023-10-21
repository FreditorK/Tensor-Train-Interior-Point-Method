import os
import sys

sys.path.append(os.getcwd() + '/../')
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from time import time
from src.utils import *
from src.optimiser import ILPSolver

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--stats", help="increase output verbosity", action="store_true")
args = parser.parse_args()

const_space = ConstraintSpace()
number_of_nodes = int(np.log2(8))
atoms = const_space.generate_atoms(number_of_nodes) # i.e. graph with 2^3 nodes
h = const_space.Hypothesis()
graph_edges = [(0, 1), (1, 7), (7, 2), (7, 6), (2, 3), (2, 4), (3, 4), (3, 5), (5, 6)]
graph = tt_svd(-2 * tt_graph_to_tensor(number_of_nodes, graph_edges) + 1)
bool_graph = tt_walsh_op_inv(graph)


def maxcut_objective(tt_train):
    return tt_inner_prod(bool_graph, tt_kronecker_prod(tt_train, tt_train))


opt = ILPSolver(const_space, objective=maxcut_objective)

if args.stats:
    number_of_edges_in_cut = []
    for i in range(100):
        print(f"---Trial {i}---")
        opt.solve()
        set_partition = tt_to_tensor(tt_walsh_op(h.value)).flatten()
        cut_edges = [e for e in graph_edges if set_partition[e[0]] * set_partition[e[1]] < 0]
        number_of_edges_in_cut.append(len(cut_edges))
    mean = np.mean(number_of_edges_in_cut)
    std = np.std(number_of_edges_in_cut)
    print(f"The mean cut edges are: {mean}, Standard deviation: {std}, Average integrality gap: {mean / 7}")
else:
    t_1 = time()
    opt.solve()
    t_2 = time()
    set_partition = tt_to_tensor(tt_walsh_op(h.value)).flatten()
    cut_edges = [e for e in graph_edges if set_partition[e[0]] * set_partition[e[1]] < 0]
    print("The cut edges are: ", cut_edges)
    G = nx.Graph()
    G.add_edges_from(graph_edges)
    pos = nx.spring_layout(G)
    edge_colors = ['red' if edge in cut_edges else 'black' for edge in G.edges()]
    nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color=edge_colors, width=2)
    plt.show()
