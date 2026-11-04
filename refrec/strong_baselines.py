"""Trainable components for the external comparison baselines.

The implementations deliberately keep the frozen sequential backbone separate:
PRM re-ranks a finite Base candidate list, while ReCODE adds a continuous-time
repeat score only when a candidate has occurred in the prefix.  The latter
matches the public SIGIR'24 implementation's five gaps, 16-dimensional latent
state, two-layer ODE function, Euler solver and summed decoder outputs.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PRMReranker(nn.Module):
    """PyTorch adaptation of PRM's personalized Transformer list re-ranker."""

    def __init__(self, emb_dim: int, list_size: int = 100, hidden_dim: int = 64,
                 num_heads: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.list_size = list_size
        self.item_projection = nn.Linear(emb_dim, hidden_dim)
        self.user_projection = nn.Linear(emb_dim, hidden_dim)
        self.dense_projection = nn.Linear(2, hidden_dim)
        self.position = nn.Embedding(list_size, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=2 * hidden_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, 1))
        self.log_residual_scale = nn.Parameter(torch.tensor(math.log(0.1)))

    def forward(self, user: torch.Tensor, item: torch.Tensor,
                base_score: torch.Tensor, history_flag: torch.Tensor) -> torch.Tensor:
        k = item.size(1)
        if k > self.list_size:
            raise ValueError(f"candidate list {k} exceeds configured size {self.list_size}")
        base_norm = (base_score - base_score.mean(1, keepdim=True)) / (
            base_score.std(1, keepdim=True).clamp_min(1e-6))
        dense = torch.stack((base_norm, history_flag.float()), dim=-1)
        pos = torch.arange(k, device=item.device)[None, :]
        x = (self.item_projection(item) + self.user_projection(user)[:, None, :] +
             self.dense_projection(dense) + self.position(pos))
        residual = self.output(self.encoder(x)).squeeze(-1)
        return base_score + self.log_residual_scale.exp().clamp_max(10.0) * residual


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 16))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ReCODEHead(nn.Module):
    """Official-style Euler neural ODE repeat-intention head.

    ``gap_bins`` is [B,K,5].  Zero denotes no previous occurrence and must
    contribute exactly zero, as in the public implementation's prepended zero.
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 64, steps: int = 51) -> None:
        super().__init__()
        self.steps = steps
        self.initial = nn.Sequential(nn.Linear(2 * emb_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, 16))
        self.ode = ODEFunc(hidden_dim)
        self.decoder = nn.Sequential(nn.Linear(16, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, 1))

    def forward(self, user: torch.Tensor, item: torch.Tensor,
                gap_bins: torch.Tensor) -> torch.Tensor:
        bsz, count, _ = item.shape
        flat_gaps = gap_bins.reshape(-1, gap_bins.size(-1)).long().clamp(0, self.steps - 1)
        active = flat_gaps.max(1).values > 0
        result = item.new_zeros(bsz * count)
        if not active.any():
            return result.view(bsz, count)
        user_flat = user[:, None, :].expand(-1, count, -1).reshape(-1, user.size(-1))[active]
        item_flat = item.reshape(-1, item.size(-1))[active]
        gaps = flat_gaps[active]
        state = torch.nn.functional.normalize(
            self.initial(torch.cat((user_flat, item_flat), dim=-1)), dim=-1)
        values = state.new_zeros(state.size(0), gaps.size(1))
        max_step = int(gaps.max().item())
        # Unit-step explicit Euler, the solver used by the released ReCODE code.
        for step in range(1, max_step + 1):
            state = state + self.ode(state)
            decoded = self.decoder(state).squeeze(-1)
            values = torch.where(gaps == step, decoded[:, None], values)
        result[active] = values.sum(1)
        return result.view(bsz, count)


def recent_gap_bins(history_items: list[int], history_times: list[int],
                    candidate_items: list[int], target_time: int,
                    time_scalar: int, steps: int, max_occurrences: int = 5) -> torch.Tensor:
    """Return target-independent time-gap bins for a candidate list."""
    occurrences: dict[int, list[int]] = {}
    for item, timestamp in zip(history_items, history_times):
        occurrences.setdefault(int(item), []).append(int(timestamp))
    out = torch.zeros(len(candidate_items), max_occurrences, dtype=torch.long)
    for row, item in enumerate(candidate_items):
        prior = occurrences.get(int(item), ())
        for col, timestamp in enumerate(reversed(prior[-max_occurrences:])):
            gap = max(1, int(math.ceil(max(0, target_time - timestamp) / max(1, time_scalar))))
            out[row, col] = min(gap, steps - 1)
    return out
