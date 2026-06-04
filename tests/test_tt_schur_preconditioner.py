import unittest

import numpy as np

from src.tt_als import cached_einsum
from src.tt_ipm import (
    _SchurCJacobiPreconditioner,
    _SchurFrameJacobiPreconditioner,
    _central_projected_block,
    _left_projected_block,
    _right_projected_block,
    _schur_preconditioner_mode,
    _should_run_local_preconditioner_benchmark,
)


class TTSchurPreconditionerTests(unittest.TestCase):
    def test_central_projected_block_matches_dense_diagonal_slices(self):
        rng = np.random.default_rng(7)
        r, ra1, n, ra2, R = 2, 3, 2, 4, 3
        X_left = rng.standard_normal((r, ra1, r))
        A_core = rng.standard_normal((ra1, n, n, ra2))
        X_right = rng.standard_normal((R, ra2, R))

        central = _central_projected_block(X_left, A_core, X_right)
        full = cached_einsum("lsr,smnS,LSR->lmLrnR", X_left, A_core, X_right).reshape(r * n * R, r * n * R)

        for i in range(r):
            for j in range(R):
                rows = [(i * n + m) * R + j for m in range(n)]
                cols = [(i * n + q) * R + j for q in range(n)]
                expected = full[np.ix_(rows, cols)]
                np.testing.assert_allclose(central[i, :, :, j], expected)


    def test_left_and_right_projected_blocks_match_dense_slices(self):
        rng = np.random.default_rng(11)
        r, ra1, n, ra2, R = 2, 3, 2, 4, 3
        X_left = rng.standard_normal((r, ra1, r))
        A_core = rng.standard_normal((ra1, n, n, ra2))
        X_right = rng.standard_normal((R, ra2, R))
        full = cached_einsum("lsr,smnS,LSR->lmLrnR", X_left, A_core, X_right).reshape(r * n * R, r * n * R)

        left = _left_projected_block(X_left, A_core, X_right)
        for j in range(R):
            rows = [(i * n + m) * R + j for i in range(r) for m in range(n)]
            np.testing.assert_allclose(left[j], full[np.ix_(rows, rows)])

        right = _right_projected_block(X_left, A_core, X_right)
        for i in range(r):
            rows = [(i * n + m) * R + j for m in range(n) for j in range(R)]
            np.testing.assert_allclose(right[i], full[np.ix_(rows, rows)])

    def test_schur_cjacobi_apply_solves_one_uncoupled_slice(self):
        n = 2
        frame = np.ones((1, 1, 1), dtype=np.float64)

        KY = np.array([[2.0, 0.1], [0.2, 1.7]])
        Aeq = np.array([[0.7, -0.2], [0.3, 0.9]])
        LZ = np.array([[2.4, 0.1], [0.2, 2.1]])
        LX = np.array([[1.1, 0.2], [0.0, 0.8]])
        Aineq = np.array([[0.5, 0.1], [-0.4, 0.6]])
        KT = np.array([[1.3, 0.2], [0.1, 1.4]])
        inv_I = np.array([[[0.8], [1.2]]])

        def core(mat):
            return mat.reshape(1, n, n, 1)

        prec = _SchurCJacobiPreconditioner(
            frame, frame, frame, frame, frame, frame,
            core(KY), core(Aeq), core(LZ), core(LX), core(Aineq), core(KT),
            frame, frame, frame, frame, frame, frame,
            inv_I,
            jitter=1e-12,
        )
        residual = np.array([1.0, -0.5, 0.2, 0.4, -0.3, 0.7])
        solution = prec.apply(residual)
        y, x, t = solution.reshape(3, 1, n, 1)[:, 0, :, 0]
        direct_mat = np.block([
            [KY, Aeq, np.zeros((n, n))],
            [-LX @ np.diag(inv_I[0, :, 0]) @ Aeq.T, LZ, -LX],
            [np.zeros((n, n)), Aineq, KT],
        ])
        np.testing.assert_allclose(direct_mat @ np.concatenate([y, x, t]), residual, rtol=1e-10, atol=1e-10)


    def test_l_and_r_modes_preserve_shape_and_finiteness(self):
        rng = np.random.default_rng(19)
        r, n, R = 2, 2, 3
        frame_left = np.zeros((r, 1, r), dtype=np.float64)
        frame_right = np.zeros((R, 1, R), dtype=np.float64)
        for i in range(r):
            frame_left[i, 0, i] = 1.0
        for j in range(R):
            frame_right[j, 0, j] = 1.0

        def core(scale):
            return (scale * np.eye(n)).reshape(1, n, n, 1)

        args = (
            frame_left, frame_left, frame_left, frame_left, frame_left, frame_left,
            core(1.5), core(0.2), core(2.0), core(0.7), core(0.3), core(1.8),
            frame_right, frame_right, frame_right, frame_right, frame_right, frame_right,
            np.ones((r, n, R), dtype=np.float64),
        )
        residual = rng.standard_normal(3 * r * n * R)
        for mode in ("l", "r"):
            prec = _SchurFrameJacobiPreconditioner(*args, mode=mode, jitter=1e-12)
            applied = prec.apply(residual)
            self.assertEqual(applied.shape, residual.shape)
            self.assertTrue(np.all(np.isfinite(applied)))


    def test_schur_rljacobi_selects_fresh_sweep_environment(self):
        self.assertEqual(_schur_preconditioner_mode("schur_rljacobi", "left_to_right"), "l")
        self.assertEqual(_schur_preconditioner_mode("schur_rljacobi", "right_to_left"), "r")
        self.assertEqual(_schur_preconditioner_mode("schur_ljacobi", "right_to_left"), "l")
        self.assertEqual(_schur_preconditioner_mode("schur_rjacobi", "left_to_right"), "r")
        self.assertIsNone(_schur_preconditioner_mode("none", "left_to_right"))
        with self.assertRaises(ValueError):
            _schur_preconditioner_mode("schur_rljacobi", None)


    def test_local_preconditioner_benchmark_filters_and_limit(self):
        ctx = {
            "enabled": True,
            "limit": 1,
            "count": 0,
            "min_res": 0.1,
            "last_iter_only": True,
            "is_last_iter": False,
            "target_iteration": 12,
            "ipm_iteration": 12,
        }
        self.assertFalse(_should_run_local_preconditioner_benchmark(ctx, 1.0))
        self.assertEqual(ctx["count"], 0)
        ctx["is_last_iter"] = True
        self.assertFalse(_should_run_local_preconditioner_benchmark(ctx, 0.01))
        self.assertEqual(ctx["count"], 0)
        self.assertTrue(_should_run_local_preconditioner_benchmark(ctx, 1.0))
        self.assertEqual(ctx["count"], 1)
        self.assertFalse(_should_run_local_preconditioner_benchmark(ctx, 1.0))


    def test_schur_cjacobi_falls_back_to_identity_for_bad_slice(self):
        n = 2
        frame = np.ones((1, 1, 1), dtype=np.float64)
        good = np.eye(n).reshape(1, n, n, 1)
        bad = np.full((1, n, n, 1), np.nan)
        inv_I = np.ones((1, n, 1), dtype=np.float64)

        prec = _SchurCJacobiPreconditioner(
            frame, frame, frame, frame, frame, frame,
            bad, good, good, good, good, good,
            frame, frame, frame, frame, frame, frame,
            inv_I,
            jitter=1e-12,
        )
        residual = np.arange(3 * n, dtype=np.float64)
        np.testing.assert_allclose(prec.apply(residual), residual)
        self.assertEqual(prec.fallback_count, 1)


if __name__ == "__main__":
    unittest.main()
