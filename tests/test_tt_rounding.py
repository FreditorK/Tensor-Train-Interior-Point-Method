import unittest

import numpy as np

from src.tt_ops import (
    tt_add,
    tt_identity,
    tt_mask_rank_reduce,
    tt_matrix_svd,
    tt_matrix_to_matrix,
    tt_norm,
    tt_one_matrix,
    tt_psd_rank_reduce,
    tt_rank_reduce,
    tt_ranks,
    tt_scale,
    tt_sub,
)


def copy_tt(train_tt):
    return [np.array(core, copy=True) for core in train_tt]


def rank_cost(train_tt):
    return sum(tt_ranks(train_tt))


class TTRoundingTests(unittest.TestCase):
    def test_rank_reduce_reports_bounded_error_when_requested(self):
        tensor = tt_add(tt_identity(3), tt_scale(1e-6, tt_one_matrix(3)))
        rounded, info = tt_rank_reduce(copy_tt(tensor), 1e-4, return_info=True)

        self.assertLessEqual(info.truncation_error, 1.01e-4)
        self.assertLessEqual(tt_norm(tt_sub(tensor, rounded)), 1.01e-4)

    def test_psd_rounding_respects_budget_and_rank_guard(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((4, 4))
        matrix = A.T @ A + 0.1 * np.eye(4)
        tensor = tt_add(tt_matrix_svd(matrix, 1e-14), tt_scale(1e-7, tt_one_matrix(2)))

        rounded, info = tt_psd_rank_reduce(copy_tt(tensor), 1e-2, return_info=True)
        rounded_matrix = tt_matrix_to_matrix(rounded)
        rounded_matrix = 0.5 * (rounded_matrix + rounded_matrix.T)

        self.assertLessEqual(tt_norm(tt_sub(tensor, rounded)), 1.01e-2)
        self.assertGreaterEqual(np.linalg.eigvalsh(rounded_matrix).min(), -1e-8)
        self.assertLessEqual(rank_cost(rounded), rank_cost(tensor))
        self.assertLessEqual(sum(info.ranks_after), sum(info.ranks_before))

    def test_mask_rounding_respects_budget_and_rank_guard(self):
        tensor = tt_add(tt_identity(3), tt_scale(1e-6, tt_one_matrix(3)))
        rounded, info = tt_mask_rank_reduce(copy_tt(tensor), tt_identity(3), 1e-3, return_info=True)

        self.assertLessEqual(tt_norm(tt_sub(tensor, rounded)), 1.01e-3)
        self.assertLessEqual(rank_cost(rounded), rank_cost(tensor))
        self.assertLessEqual(sum(info.ranks_after), sum(info.ranks_before))


if __name__ == "__main__":
    unittest.main()
