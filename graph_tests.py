import numpy as np
import jax.numpy as jnp
from operators import D_func

graph = -2.0*np.array([
    [0, -1, 0, -1, 0, 0],
    [-1, 0, -1, 0, -1, -1],
    [0, -1, 0, -1, -1, 0],
    [-1, 0, -1, 0, 0, -1],
    [0, -1, -1, 0, 0, 0],
    [0, -1, 0, -1, 0, 0]
], dtype=float)-1

graph_2 = -2.0*np.array([
    [0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, -1, 0],
    [0, 0, 0, -1, 0, 0],
    [0, 0, -1, 0, 0, 0],
    [0, -1, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0]
])-1


def loss(M):
    eigs, _ = jnp.linalg.eigh(M)
    return sum([(eigs[i] - eigs[i+1])**2 for i in range(len(eigs)-1)])


assert np.all(graph == graph.T), "Is not symmetric!"
eigs, eig_v = np.linalg.eigh(graph)
print(eigs)
print(np.round(eig_v, decimals=3))
"""
grad = D_func(loss)
for _ in range(100):
    graph -= 0.1*grad(graph)
print(np.round(graph, decimals=2))
eigs, eig_v = np.linalg.eigh(graph)
print(eigs)
print(np.round(eig_v, decimals=3))
"""
eigs_2, eig_v2 = np.linalg.eigh(graph_2)
print(eigs_2)
print(np.round(eig_v2, decimals=3))

eigs_3, eig_v3 = np.linalg.eigh(graph - graph_2 + graph*graph_2)
print(eigs_3)
print(np.round(eig_v3, decimals=3))

print(0.5 + 0.5*graph + 0.5*graph_2 - 0.5*graph * graph_2)
print(0.5 + 0.5*eigs + 0.5*eigs_2 - 0.5*eigs*eigs_2)

