import numpy as np

A = np.array([[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, -1, 0, 1]])

A_op = np.diag(A.flatten())
one = np.ones((4, 4)).reshape(-1, 1)
print((A_op @ one).reshape(4, 4))
A_op_compl = np.diag((A == 0).flatten())

block_system = np.block([[np.eye(len(A_op)), A_op],
                         [A_op.T, A_op_compl]])

b = np.block([[(A != 0).reshape(-1, 1)], [np.zeros((16, 1))]])
sol = np.linalg.solve(block_system, b)
print(sol[:16].reshape(4, 4))
print(sol[16:].reshape(4, 4))
print((A_op.T @ (A != 0).reshape(-1, 1)).reshape(4, 4))