"""Mamba4Rec backbone under the shared full-ranking RECal protocol.

The encoder follows the public Mamba4Rec implementation while reusing this
project's data split, sampled training objective, candidate set, and evaluator.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from refrec.seqrec import SeqRecBase


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * dim, dim), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x) + x)


class MambaLayer(nn.Module):
    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int,
                 dropout: float, residual: bool) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:  # pragma: no cover - depends on CUDA wheel
            raise RuntimeError(
                "Mamba4Rec requires mamba-ssm; use the isolated .venv_faes environment"
            ) from exc
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        # The unfused public path is compatible with the installed Torch/CUDA
        # wheels and implements the same linear-time selective scan.
        if hasattr(self.mamba, "use_fast_path"):
            self.mamba.use_fast_path = False
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim, eps=1e-12)
        self.ffn = FeedForward(dim, dropout)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(self.mamba(x))
        y = self.norm(y + x) if self.residual else self.norm(y)
        return self.ffn(y)


class Mamba4RecBackbone(SeqRecBase):
    """Linear-time Mamba4Rec encoder with the common candidate scorer."""

    def __init__(self, *args, mamba_layers: int = 1, mamba_d_state: int = 32,
                 mamba_d_conv: int = 4, mamba_expand: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if mamba_layers < 1:
            raise ValueError("mamba_layers must be positive")
        dropout = float(kwargs.get("dropout", 0.1))
        self.input_norm = nn.LayerNorm(self.emb_dim, eps=1e-12)
        self.layers = nn.ModuleList([
            MambaLayer(self.emb_dim, mamba_d_state, mamba_d_conv, mamba_expand,
                       dropout, residual=mamba_layers > 1)
            for _ in range(mamba_layers)
        ])
        self._reset_mamba_parameters()

    def _reset_mamba_parameters(self) -> None:
        # Match the initialization used by the public implementation without
        # overwriting Mamba's specialized state-space parameters.
        for module in (self.input_norm, self.layers):
            for child in module.modules():
                if isinstance(child, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(child.weight, mean=0.0, std=0.02)
                    if isinstance(child, nn.Linear) and child.bias is not None:
                        nn.init.zeros_(child.bias)
                elif isinstance(child, nn.LayerNorm):
                    nn.init.ones_(child.weight)
                    nn.init.zeros_(child.bias)

    def item_repr(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(item_ids)

    def compute_base_score(self, enc, seq_items, valid_mask, cand_items):
        del seq_items, valid_mask
        return (enc["h_user"].unsqueeze(1) * self.item_embedding(cand_items)).sum(-1)

    def encode_sequence(self, seq_items, seq_cats, valid_mask, recent_mask):
        del seq_cats, recent_mask
        x = self.emb_dropout(self.input_norm(self.item_embedding(seq_items)))
        x = x * valid_mask.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x)
            x = x * valid_mask.unsqueeze(-1)
        return {"h_user": self._last_hidden(x, valid_mask)}

    def complexity_note(self, num_neg: int = 100) -> str:
        del num_neg
        return ("Mamba4Rec linear-time selective-state encoder; full-catalog decoding "
                "is chunked; no sequence-length quadratic term")
