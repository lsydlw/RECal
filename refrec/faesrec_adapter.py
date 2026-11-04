"""Adapter for the official FAESRec encoder under the shared RECal protocol.

The upstream implementation is vendored separately.  This module deliberately
reuses only its encoder and frequency augmentation layers; data splitting,
negative sampling, validation selection and full-catalog evaluation remain the
same as for the other backbones in :mod:`refrec.run_seqrec`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from refrec.seqrec import SeqRecBase


def _load_upstream():
    root = Path(__file__).resolve().parent.parent / "third_party" / "FAESRec"
    if not root.exists():
        raise RuntimeError(f"FAESRec upstream checkout is missing: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from recbole.model.sequential_recommender.faesrec import Encoder, FreqAug
    return Encoder, FreqAug


class FAESRecBackbone(SeqRecBase):
    """Official frequency-adaptive encoder with the project's scoring API."""

    def __init__(
        self,
        *args,
        num_blocks: int = 2,
        num_heads: int = 2,
        faes_expert_num: int = 4,
        faes_temperature: float = 1.0,
        faes_cl_weight: float = 1e-2,
        faes_l1_weight: float = 1e-3,
        faes_d_state: int = 32,
        faes_d_conv: int = 4,
        faes_expand: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        Encoder, FreqAug = _load_upstream()
        config = {
            "n_layers": num_blocks,
            "n_heads": num_heads,
            "hidden_size": self.emb_dim,
            "inner_size": self.emb_dim * 4,
            "hidden_dropout_prob": float(kwargs.get("dropout", 0.1)),
            "attn_dropout_prob": float(kwargs.get("dropout", 0.1)),
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-12,
            "initializer_range": 0.02,
            "expert_num": faes_expert_num,
            "temperature": faes_temperature,
            "l1_weight": faes_l1_weight,
            "cl_weight": faes_cl_weight,
            "d_state": faes_d_state,
            "d_conv": faes_d_conv,
            "expand": faes_expand,
        }
        self.position_embedding = nn.Embedding(self.max_seq_len, self.emb_dim)
        self.input_norm = nn.LayerNorm(self.emb_dim, eps=1e-12)
        self.encoder = Encoder(config)
        # FAESRec pins mamba-ssm 2.2.3.post2 together with causal-conv1d
        # 1.5.2.  Their optional fused path uses incompatible low-level CUDA
        # signatures on the Torch 2.6 wheels.  The standard Mamba path is the
        # same linear-time operator and uses causal-conv1d's public autograd
        # interface, so disable only the incompatible fusion optimization.
        for module in self.encoder.modules():
            if hasattr(module, "use_fast_path"):
                module.use_fast_path = False
        self.freq_aug = FreqAug(self.max_seq_len, config)
        self.faes_temperature = float(faes_temperature)
        self.faes_cl_weight = float(faes_cl_weight)
        self.faes_l1_weight = float(faes_l1_weight)
        self._reset_faes_parameters()

    def _reset_faes_parameters(self) -> None:
        for module in (self.position_embedding, self.input_norm, self.encoder):
            for child in module.modules():
                if isinstance(child, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(child.weight, mean=0.0, std=0.02)
                    if isinstance(child, nn.Linear) and child.bias is not None:
                        nn.init.zeros_(child.bias)
                elif isinstance(child, nn.LayerNorm):
                    nn.init.ones_(child.weight)
                    nn.init.zeros_(child.bias)

    def item_repr(self, item_ids: torch.Tensor) -> torch.Tensor:
        # The published FAESRec decoder uses the item embedding only.
        return self.item_embedding(item_ids)

    def compute_base_score(self, enc, seq_items, valid_mask, cand_items):
        return (enc["h_user"].unsqueeze(1) * self.item_embedding(cand_items)).sum(dim=-1)

    def _input_embedding(self, seq_items: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        bsz, length = seq_items.shape
        positions = torch.arange(length, device=seq_items.device).unsqueeze(0).expand(bsz, -1)
        x = self.item_embedding(seq_items) + self.position_embedding(positions)
        x = self.emb_dropout(self.input_norm(x))
        return x * valid_mask.unsqueeze(-1)

    def _encode_tensor(self, seq_items: torch.Tensor, valid_mask: torch.Tensor, x: torch.Tensor):
        hidden = self.encoder(seq_items, x)
        return self._last_hidden(hidden, valid_mask)

    def encode_sequence(self, seq_items, seq_cats, valid_mask, recent_mask):
        del seq_cats, recent_mask
        x = self._input_embedding(seq_items, valid_mask)
        return {"h_user": self._encode_tensor(seq_items, valid_mask, x), "faes_input": x}

    def _contrastive_loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        batch = a.size(0)
        z = torch.cat([a, b], dim=0)
        similarity = torch.mm(z, z.t()) / self.faes_temperature
        positive = torch.cat([
            torch.diag(similarity, batch), torch.diag(similarity, -batch)
        ]).reshape(2 * batch, 1)
        mask = torch.ones((2 * batch, 2 * batch), device=z.device, dtype=torch.bool)
        mask.fill_diagonal_(False)
        idx = torch.arange(batch, device=z.device)
        mask[idx, idx + batch] = False
        mask[idx + batch, idx] = False
        negative = similarity[mask].reshape(2 * batch, -1)
        logits = torch.cat([positive, negative], dim=1)
        labels = torch.zeros(2 * batch, device=z.device, dtype=torch.long)
        return F.cross_entropy(logits, labels)

    def forward(self, seq_items, seq_cats, valid_mask, weak_mask, recent_mask, sampled_items):
        enc = self.encode(seq_items, seq_cats, valid_mask, weak_mask, recent_mask)
        out = self.score_candidates(enc, seq_items, valid_mask, sampled_items)
        if self.training:
            augmented = self.freq_aug(enc["faes_input"])
            aug_user = self._encode_tensor(seq_items, valid_mask, augmented)
            cl = self._contrastive_loss(enc["h_user"], aug_user)
            sparse = torch.norm(self.freq_aug.para[:, 0], p=1) / self.max_seq_len
            out["backbone_aux_loss"] = self.faes_cl_weight * cl + self.faes_l1_weight * sparse
        return out

    def complexity_note(self, num_neg: int = 100) -> str:
        return ("FAESRec frequency-adaptive encoder; full-catalog decoding is chunked; "
                "RECal-compatible score interface preserves all history candidates")
