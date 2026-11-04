import unittest

import torch

from refrec.run_comparison_baselines import calibrated_top20
from refrec.strong_baselines import PRMReranker, ReCODEHead, recent_gap_bins


class ComparisonBaselineTests(unittest.TestCase):
    def test_recode_unseen_candidates_have_exactly_zero_residual(self):
        head = ReCODEHead(emb_dim=8, steps=8)
        user = torch.randn(2, 8)
        item = torch.randn(2, 3, 8)
        gaps = torch.zeros(2, 3, 5, dtype=torch.long)
        self.assertTrue(torch.equal(head(user, item, gaps), torch.zeros(2, 3)))

    def test_gap_features_use_prefix_only(self):
        bins = recent_gap_bins([7, 8, 7], [10, 20, 30], [7, 9], 50,
                               time_scalar=10, steps=20)
        self.assertEqual(bins[0, :2].tolist(), [2, 4])
        self.assertEqual(bins[1].sum().item(), 0)

    def test_prm_accepts_training_and_evaluation_list_lengths(self):
        model = PRMReranker(emb_dim=8, list_size=10, hidden_dim=8)
        for length in (5, 10):
            score = model(torch.randn(2, 8), torch.randn(2, length, 8),
                          torch.randn(2, length), torch.zeros(2, length))
            self.assertEqual(score.shape, (2, length))

    def test_calibrated_reranking_never_invents_candidates(self):
        items = [1, 2, 3, 4]
        result = calibrated_top20(items, [4.0, 3.0, 2.0, 1.0], {1, 3},
                                  {1: 1, 2: 2, 3: 1, 4: 2}, 2, 0.2)
        self.assertEqual(set(result), set(items))
        self.assertEqual(len(result), len(set(result)))


if __name__ == "__main__":
    unittest.main()
