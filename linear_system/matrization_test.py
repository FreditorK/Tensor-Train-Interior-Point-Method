import sys
import os

sys.path.append(os.getcwd() + '/../')
from src.tt_ops import *


np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=4, suppress=True)
s_matrix_tt = tt_random_gaussian([4, 4], shape=(2, 2))
s_matrix_tt_2 = tt_random_gaussian([4, 4], shape=(2, 2))
s_vec = tt_random_gaussian([4, 4], shape=(2,))


print(np.round(tt_matrix_to_matrix(tt_mat_mat_mul(s_matrix_tt, s_matrix_tt_2)), decimals=3))

print(np.round(tt_matrix_to_matrix(s_matrix_tt) @ tt_matrix_to_matrix(s_matrix_tt_2), decimals=3))
vec_s_matrix_tt = tt_vec(s_matrix_tt)
identity = tt_identity(3) #tt_identity(4)
#print(np.round(tt_matrix_to_matrix(s_matrix_tt_2), decimals=2))
s_matrix_tt_2 = tt_kron(tt_transpose(s_matrix_tt_2), identity)
#print(np.round(tt_matrix_to_matrix(s_matrix_tt_2), decimals=2))
result = tt_matrix_vec_mul(s_matrix_tt_2, vec_s_matrix_tt)
mat_result = tt_mat(result)
print(np.round(tt_matrix_to_matrix(mat_result), decimals=3))
ttmata = tt_matrix_svd(tt_matrix_to_matrix(mat_result))
print(np.round(tt_matrix_to_matrix(ttmata), decimals=3))


#TODO: Implement unit tests for all operations

#print(np.round(tt_matrix_to_matrix(s_matrix_tt_2) @ tt_vec_to_vec(vec_s_matrix_tt), decimals=3))