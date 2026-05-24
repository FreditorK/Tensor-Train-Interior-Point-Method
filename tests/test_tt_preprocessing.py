import unittest

import numpy as np

from src.tt_ipm import LGMRESSolver, _preprocess_problem_data
from src.tt_ops import (
    tt_add,
    tt_diag,
    tt_diag_op,
    tt_diagonal,
    tt_identity,
    tt_inner_prod,
    tt_norm,
    tt_one_matrix,
    tt_rank_reduce,
    tt_reshape,
    tt_scale,
    tt_sub,
    tt_transpose,
    tt_fast_matrix_vec_mul,
)


class TTPreprocessingTests(unittest.TestCase):
    def test_petsc_lgmres_shell_uses_read_locked_input(self):
        class DenseOp:
            def matvec(self, x):
                return np.array([4.0 * x[0] + x[1], x[0] + 3.0 * x[1]])

        solver = LGMRESSolver(rtol=1e-12, restart=10, outer_k=3)
        try:
            sol = solver.solve_system(DenseOp(), np.array([1.0, 2.0]), (2, 2))
        finally:
            solver.destroy()

        self.assertLess(np.linalg.norm(np.array([[4.0, 1.0], [1.0, 3.0]]) @ sol - [1.0, 2.0]), 1e-10)

    def test_diag_objective_projection_is_stored_and_removed(self):
        dim = 3
        identity = tt_identity(dim)
        offdiag = tt_sub(tt_one_matrix(dim), identity)
        obj_tt = tt_reshape(tt_add(identity, tt_scale(0.25, offdiag)), (4,))
        lin_op_tt = tt_diag_op(identity)
        bias_tt = tt_reshape(identity, (4,))
        lag_maps = {"y": tt_diag_op(offdiag)}

        _, obj_p, lin_p, bias_p, _, info = _preprocess_problem_data(
            lag_maps, obj_tt, lin_op_tt, bias_tt, None, 1e-8, 1e-12
        )

        diag_p = tt_diag(tt_diagonal(tt_reshape(obj_p, (2, 2))))
        self.assertTrue(info.objective_projected)
        self.assertAlmostEqual(info.objective_offset, float(2 ** dim), places=8)
        self.assertLess(tt_norm(diag_p), 1e-7)
        self.assertGreater(info.constraint_scale, 0.0)
        self.assertGreater(info.psd_scale, 0.0)
        self.assertGreater(tt_norm(lin_p), 0.0)
        self.assertGreater(tt_norm(bias_p), 0.0)

    def test_projected_dual_offset_restores_original_dual_objective(self):
        dim = 2
        identity = tt_identity(dim)
        obj_tt = tt_reshape(identity, (4,))
        lin_op_tt = tt_diag_op(identity)
        bias_tt = tt_reshape(identity, (4,))
        lag_maps = {"y": tt_diag_op(tt_sub(tt_one_matrix(dim), identity))}

        _, obj_p, _, _, _, info = _preprocess_problem_data(
            lag_maps, obj_tt, lin_op_tt, bias_tt, None, 1e-8, 1e-12
        )
        restored = tt_rank_reduce(
            tt_add(
                tt_scale(1.0 / info.psd_scale ** 2, obj_p),
                tt_fast_matrix_vec_mul(tt_transpose(lin_op_tt), info.y_offset_tt, 1e-12),
            ),
            1e-12,
        )

        self.assertLess(tt_norm(tt_sub(restored, obj_tt)), 5e-8)
        self.assertAlmostEqual(tt_inner_prod(bias_tt, info.y_offset_tt), info.objective_offset, places=8)


if __name__ == "__main__":
    unittest.main()
