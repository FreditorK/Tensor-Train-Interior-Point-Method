import sys
import os
import numpy as np

sys.path.append(os.getcwd() + '/../')
from src.tt_ipm import LGMRESSolver

import petsc4py
print(petsc4py.__version__)



class MyMatVec:
    def matvec(self, x):
        # Example: A is implicit (e.g., x * 2)
        return 2.0 * x

rhs = np.ones(100)
solver = LGMRESSolver()
solution = solver.solve_system(MyMatVec(), rhs, (100, 100))
print(solution)  # Should be 0.5 * ones(100)


class MyMatVec2:
    def matvec(self, x):
        # Example: A is implicit (e.g., x * 2)
        return 5.0 * x

rhs = np.ones(10)
solver = LGMRESSolver()
solution = solver.solve_system(MyMatVec2(), rhs, (10, 10))
print(solution)  # Should be 1/5 * ones(100)
