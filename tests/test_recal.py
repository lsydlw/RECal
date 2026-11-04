import math
import unittest
import torch

from refrec.recal import (RepeatIntentHead, asymmetric_risk_trust, bidirectional_risk_trust,
                          calibrated_history_mass, entropy_confidence,
                          group_logit_shift, sequence_statistics,
                          shrink_target_mass)


class RECalTests(unittest.TestCase):
    def test_vectorized_sequence_statistics_match_definition(self):
        items = torch.tensor([[0, 0, 1, 2, 1], [0, 3, 3, 4, 5]])
        mask = (items > 0).float()
        got = sequence_statistics(items, mask, recent_window=3)
        expected = torch.tensor([
            [math.log1p(3), 2 / 3, 1 / 3, 1 / 3, 2 / 3],
            [math.log1p(4), 3 / 4, 1 / 4, 0.0, 1 / 4],
        ])
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))
    def test_context_only_intent_head_accepts_no_explicit_statistics(self):
        head = RepeatIntentHead(emb_dim=8, num_stats=0, hidden_dim=4, dropout=0.0)
        context = torch.randn(3, 8)
        empty_stats = torch.empty(3, 0)
        head.set_stats_normalization(empty_stats)
        logits = head(context, empty_stats)
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertTrue(torch.isfinite(logits).all())

    def test_closed_form_hits_requested_mass(self):
        m = torch.tensor([0.01, 0.2, 0.5, 0.9])
        q = torch.tensor([0.7, 0.1, 0.5, 0.2])
        got = calibrated_history_mass(m, group_logit_shift(q, m))
        self.assertTrue(torch.allclose(got, q, atol=1e-6))

    def test_direction_and_identity(self):
        m = torch.tensor([0.2, 0.5, 0.8])
        q = torch.tensor([0.7, 0.5, 0.1])
        d = group_logit_shift(q, m)
        self.assertGreater(float(d[0]), 0)
        self.assertAlmostEqual(float(d[1]), 0.0, places=6)
        self.assertLess(float(d[2]), 0)

    def test_statistics_do_not_take_target(self):
        seq = torch.tensor([[0, 1, 2, 1], [0, 0, 3, 4]])
        mask = seq.gt(0).float()
        a = sequence_statistics(seq, mask)
        b = sequence_statistics(seq.clone(), mask.clone())
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(tuple(a.shape), (2, 5))

    def test_shrinkage_endpoints_and_entropy_bounds(self):
        m = torch.tensor([0.2, 0.8])
        q = torch.tensor([0.7, 0.1])
        self.assertTrue(torch.allclose(shrink_target_mass(m, q, 0.0), m))
        self.assertTrue(torch.allclose(shrink_target_mass(m, q, 1.0), q))
        c = entropy_confidence(torch.tensor([1e-6, 0.5, 1 - 1e-6]))
        self.assertTrue(torch.all((c >= 0) & (c <= 1)))
        self.assertAlmostEqual(float(c[1]), 0.0, places=6)

    def test_asymmetric_trust_is_full_upward_and_conservative_downward(self):
        m = torch.tensor([0.1, 0.9, 0.9])
        q = torch.tensor([0.2, 0.0, 0.2])
        t = asymmetric_risk_trust(m, q, power=8)
        self.assertAlmostEqual(float(t[0]), 1.0, places=6)
        self.assertAlmostEqual(float(t[1]), 1.0, places=6)
        self.assertLess(float(t[2]), 0.02)

    def test_bidirectional_trust_needs_directional_confidence(self):
        m = torch.tensor([0.1, 0.1, 0.9])
        q = torch.tensor([0.2, 0.8, 0.2])
        t = bidirectional_risk_trust(m, q, down_power=12, up_power=2)
        self.assertLess(float(t[0]), float(t[1]))
        self.assertLess(float(t[2]), 0.01)


if __name__ == "__main__": unittest.main()
