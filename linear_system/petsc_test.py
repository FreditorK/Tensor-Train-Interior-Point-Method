import sys
import os
import numpy as np

sys.path.append(os.getcwd() + '/../')
from src.tt_ipm import LGMRESSolver

import petsc4py
from petsc4py import PETSc

class SaddlePointSolver:
    """
    A matrix-free solver using the robust IS-based PCFIELDSPLIT approach.
    """
    
    # The context object now holds all four operators
    class _ShellContext:
        def __init__(self, A, B, C, D):
            self.A, self.B, self.C, self.D = A, B, C, D
            self.n = D.shape[0] # Size of the first block (y-variables)

        def mult(self, _, x, y):
            # This performs the full matvec for M = [[D, C], [B, A]]
            x_np = x.getArray()
            y_np = np.zeros_like(x_np)
            
            # Split the input vector x into y_part and x_part
            y_in = x_np[:self.n]
            x_in = x_np[self.n:]
            
            # Perform the block multiplications
            # Top row: D*y_in + C*x_in
            y_np[:self.n] = self.D.matvec(y_in) + self.C.matvec(x_in)
            # Bottom row: B*y_in + A*x_in
            y_np[self.n:] = self.B.matvec(y_in) + self.A.matvec(x_in)
            
            y.setArray(y_np)
            y.assemble()

    def __init__(self):
        self.ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
        self.M = None

    def setup_solver(self, m, n):
        self.ksp.setType('fgmres')
        self.ksp.setTolerances(rtol=1e-8, max_it=100)

        pc = self.ksp.getPC()
        pc.setType('fieldsplit')
        
        # --- KEY CHANGE: Define fields using Index Sets (IS) ---
        is_y = PETSc.IS().createStride(n, 0, 1, comm=PETSc.COMM_WORLD)
        is_x = PETSc.IS().createStride(m, n, 1, comm=PETSc.COMM_WORLD)
        pc.setFieldSplitIS(("y_field", is_y), ("x_field", is_x))
        # --------------------------------------------------------

        pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.UPPER)

        self.ksp.setUp()
        sub_ksps = pc.getFieldSplitSubKSP()

        # Config for D-block (field 0, now named "y_field")
        sub_ksps[0].setType('cg')
        sub_ksps[0].setTolerances(rtol=1e-12, max_it=5)
        sub_ksps[0].getPC().setType('none')

        # Config for A-block (field 1, now named "x_field")
        sub_ksps[1].setType('lgmres')
        sub_ksps[1].setTolerances(rtol=1e-5, max_it=50)
        sub_ksps[1].setGMRESRestart(20)
        #sub_ksps[1].setLGMRESAugment(5)
        sub_ksps[1].getPC().setType('none')

        self.ksp.setFromOptions()

    def solve(self, A_op, B_op, C_op, D_op, f_vec, g_vec):
        m, n = A_op.shape[0], D_op.shape[0]
        N = m + n

        # Create the context for the single monolithic matrix
        context = self._ShellContext(A_op, B_op, C_op, D_op)
        
        # Create a single MatPython shell for the whole system
        self.M = PETSc.Mat().createPython(
            ((N, N)),
            context=context,
            comm=PETSc.COMM_WORLD
        )
        self.M.setUp()

        self.ksp.setOperators(self.M)
        self.setup_solver(m, n)
        
        # Create flat vectors for RHS and solution
        rhs_np = np.concatenate([g_vec, f_vec])
        rhs = PETSc.Vec().createWithArray(rhs_np, comm=PETSc.COMM_WORLD)
        sol_vec = self.M.createVecRight()
        
        self.ksp.solve(rhs, sol_vec)

        solution_flat = sol_vec.getArray()
        sol_y = solution_flat[:n]
        sol_x = solution_flat[n:]

        rhs.destroy()
        sol_vec.destroy()
        
        return sol_x, sol_y

    def destroy(self,_=None):
        if hasattr(self, "ksp") and self.ksp:
            self.ksp.destroy()
            self.ksp = None
        del self.M

if __name__ == '__main__':
    # Define a dummy operator class for demonstration
    class UserOperator:
        def __init__(self, shape, multiplier=1.0):
            self.shape = shape
            self.multiplier = multiplier

        def matvec(self, x):
            return x * self.multiplier

    m = 50
    A_op = UserOperator((m, m), 2.0)
    B_op = UserOperator((m, m), -1.0)
    C_op = UserOperator((m, m), 1.0)
    D_op = UserOperator((m, m), 3.0)

    f_vec = np.random.rand(m)
    g_vec = np.random.rand(m)

    solver = SaddlePointSolver()
    try:
        sol_x, sol_y = solver.solve(A_op, B_op, C_op, D_op, f_vec, g_vec)
        print("\nSolve successful.")
        print(np.linalg.norm(A_op.matvec(sol_x) + B_op.matvec(sol_y) - f_vec))
        print(np.linalg.norm(C_op.matvec(sol_x) + D_op.matvec(sol_y) - g_vec))
    finally:
        solver.destroy()