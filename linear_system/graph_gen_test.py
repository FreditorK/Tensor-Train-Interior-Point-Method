import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
import copy

#np.random.seed(53)
np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)


def generate_random_projection(basis):
    r = len(basis)
    k = np.random.randint(r)
    random_indices_1 = np.random.choice(r, size=k, replace=False)
    random_indices_2 = np.random.choice(r, size=k, replace=False)
    projector = np.eye(r)
    for i, j in zip(random_indices_1, random_indices_2):
        projector += np.outer(basis[i], basis[j] - basis[i])
    return projector

def generate_binary_matrix(r, n):
    random_matrix = np.random.randn(r, r)
    q, _ = np.linalg.qr(random_matrix, mode='reduced')
    basis_vectors = q.T

    indices = np.concatenate((np.random.choice(r, size=2, replace=False), np.random.choice(r, size=2, replace=True)))
    np.random.shuffle(indices)
    first_indices = [indices[0], indices[1], indices[2], indices[3]]
    first_core = basis_vectors[first_indices]

    train = [first_core.reshape(1, 4, r)]
    for _ in range(n-2):
        core = np.empty((r, 4, r))
        core[:,  1] = generate_random_projection(basis_vectors)
        core[:,  2] = generate_random_projection(basis_vectors)
        core[:,  0] = generate_random_projection(basis_vectors)
        core[:,  3] = generate_random_projection(basis_vectors)
        train.append(core)

    indices = np.concatenate((np.random.choice(r, size=2, replace=False), np.random.choice(r, size=2, replace=True)))
    np.random.shuffle(indices)
    last_indices = [indices[0], indices[1], indices[2], indices[3]]
    last_core = basis_vectors[last_indices]
    train.append(last_core.reshape(1, 4, r).transpose(2, 1, 0))
    return train
    

r = 4
#graph_tt = generate_binary_matrix(r, 4)
#graph_tt = tt_reshape(tt_rank_reduce_py(graph_tt, 1e-12), (2, 2))

#print(tt_matrix_to_matrix(graph_tt))
#print(tt_ranks(graph_tt))
liu = tt_random_graph(4, r) #tt_sub(tt_tril_one_matrix(4), tt_identity(4))
liu  = tt_rank_reduce_py(tt_reshape(liu, (2, 2)), 1e-12)
#liu = tt_fast_hadamard(liu, graph_tt)
#liu = tt_add(liu, tt_transpose(liu))
#liu = tt_rank_reduce_py(liu, 1e-12)
#print(tt_matrix_to_matrix(liu))
print(tt_ranks(liu))
mliu = tt_matrix_to_matrix(liu)
print(mliu)
print(np.sum(mliu)/(mliu.shape[0]*mliu.shape[1]))
"""
for i, c in enumerate(liu):
    print(f"{i}")
    print(c[:, 0])
    print(c[:, 1])
    print(c[:, 2])
    print(c[:, 3])
"""