import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_maxcut(adj_matrix, nodes_in_cut, duality_gaps):
    adj_matrix = np.round((adj_matrix + 1) / 2)
    graph = nx.from_numpy_array(adj_matrix)
    fig, axs = plt.subplots(1, 2, figsize=(10, 20))
    axs[0].set_title("MaxCut")
    axs[1].set_title("Duality Gap over iterations")
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos, ax=axs[0], with_labels=True, node_color='lightblue', edge_color='gray', node_size=300,
                     font_size=10, font_color='black')
    nx.draw_networkx_nodes(graph, pos, ax=axs[0], nodelist=nodes_in_cut, alpha=1.0, node_color='r')
    axs[1].plot(duality_gaps)
    plt.show()


def plot_duality_gaps(duality_gaps):
    fig, ax = plt.subplots(1, 1, figsize=(10, 20))
    ax.set_title("Duality Gap over iterations")
    ax.plot(duality_gaps)
    plt.show()

