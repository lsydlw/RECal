import unittest
import torch

from refrec.seqrec import RepeatNet


class RepeatNetProtocolTests(unittest.TestCase):
    def _model(self):
        torch.manual_seed(7)
        return RepeatNet(num_items=17, num_categories=3,
                         item2cat={i: i % 3 + 1 for i in range(1, 18)},
                         emb_dim=8, max_seq_len=5, dropout=0.0,
                         repeat_head="base")

    def test_full_catalog_scores_are_chunk_invariant(self):
        m = self._model().eval()
        seq = torch.tensor([[0, 1, 2, 1, 3], [0, 0, 4, 5, 6]])
        cats = torch.tensor([[0, 2, 3, 2, 1], [0, 0, 2, 3, 1]])
        mask = seq.gt(0).float()
        enc = m.encode(seq, cats, mask, mask * 0, mask)
        items = torch.arange(1, 18)
        enc = m.prepare_full_catalog(enc, seq, mask, items, chunk_size=3)
        whole = m.score_chunk(enc, seq, mask, items)
        chunks = torch.cat([m.score_chunk(enc, seq, mask, items[i:i+4])
                            for i in range(0, len(items), 4)], dim=1)
        self.assertTrue(torch.allclose(whole, chunks, atol=1e-6))

    def test_repeatnet_outputs_log_probabilities(self):
        m = self._model().eval()
        seq = torch.tensor([[0, 1, 2, 1, 3]])
        cats = torch.tensor([[0, 2, 3, 2, 1]])
        mask = seq.gt(0).float(); items = torch.arange(1, 18)
        enc = m.encode(seq, cats, mask, mask * 0, mask)
        enc = m.prepare_full_catalog(enc, seq, mask, items, chunk_size=5)
        scores = m.score_chunk(enc, seq, mask, items)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertAlmostEqual(float(torch.exp(scores).sum()), 1.0, places=5)


if __name__ == "__main__": unittest.main()
