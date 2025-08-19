import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *
import copy

#np.random.seed(53)
np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
    

r = 3
#graph_tt = generate_binary_matrix(r, 4)
#graph_tt = tt_reshape(tt_rank_reduce_py(graph_tt, 1e-12), (2, 2))

#print(tt_matrix_to_matrix(graph_tt))
#print(tt_ranks(graph_tt))
stat = np.zeros(r)
for _ in range(10):
    liu = tt_random_graph(3, r, skew=-1) #tt_sub(tt_tril_one_matrix(4), tt_identity(4))
    stat[int(np.max(tt_ranks(liu)))-1] += 1
    print(tt_matrix_to_matrix(liu))

stat = np.vstack((np.arange(1, r+1), stat))
print(stat)