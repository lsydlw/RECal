"""Repeat--explore probability-mass calibration."""

from __future__ import annotations

import torch
import torch.nn as nn


def group_logit_shift(q: torch.Tensor, m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Closed-form KL projection shift for the history candidate group."""
    q = q.clamp(eps, 1.0 - eps)
    m = m.clamp(eps, 1.0 - eps)
    return torch.logit(q) - torch.logit(m)


def calibrated_history_mass(m: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """History mass after adding ``delta`` to every history-item logit."""
    odds = (m / (1.0 - m).clamp_min(1e-30)) * torch.exp(delta)
    return odds / (1.0 + odds)


def entropy_confidence(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Confidence in a Bernoulli intent estimate, in [0, 1]."""
    p = q.clamp(eps, 1.0 - eps)
    entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
    return (1.0 - entropy / torch.log(torch.tensor(2.0, device=q.device))).clamp(0.0, 1.0)


def shrink_target_mass(m: torch.Tensor, q: torch.Tensor, trust: torch.Tensor | float) -> torch.Tensor:
    """Shrink the intent target toward the base mass when trust is below one."""
    if not isinstance(trust, torch.Tensor):
        trust = torch.as_tensor(trust, dtype=m.dtype, device=m.device)
    return m + trust * (q - m)


def asymmetric_risk_trust(m: torch.Tensor, q: torch.Tensor, power: float = 8.0) -> torch.Tensor:
    """Conservative trust for downward corrections in mixed-intent regimes.

    Upward corrections retain full trust.  A downward correction is strong only
    when repeat intent is very close to zero; this limits false suppression of
    genuinely repeated targets while preserving pure-explore behavior.
    """
    explore_certainty = (1.0 - 2.0 * q).clamp(0.0, 1.0).pow(power)
    return torch.where(q >= m, torch.ones_like(q), explore_certainty)


def bidirectional_risk_trust(m: torch.Tensor, q: torch.Tensor,
                             down_power: float = 12.0,
                             up_power: float = 2.0) -> torch.Tensor:
    """Risk-aware trust requiring intent confidence in both directions."""
    down = (1.0 - 2.0 * q).clamp(0.0, 1.0).pow(down_power)
    up = (2.0 * q).clamp(0.0, 1.0).pow(up_power)
    return torch.where(q >= m, up, down)


class RepeatIntentHead(nn.Module):
    """Small target-independent predictor of next-event repeat intent."""

    def __init__(self, emb_dim: int, num_stats: int = 5, hidden_dim: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.context_norm = nn.LayerNorm(emb_dim)
        self.net = nn.Sequential(
            nn.Linear(emb_dim + num_stats, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("stats_mean", torch.zeros(num_stats))
        self.register_buffer("stats_std", torch.ones(num_stats))

    def set_stats_normalization(self, stats: torch.Tensor) -> None:
        if stats.shape[1] == 0:
            return
        self.stats_mean.copy_(stats.mean(dim=0))
        self.stats_std.copy_(stats.std(dim=0, unbiased=False).clamp_min(1e-6))

    def forward(self, context: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        z = (stats - self.stats_mean) / self.stats_std
        return self.net(torch.cat([self.context_norm(context), z], dim=-1)).squeeze(-1)


class MultiGroupIntentHead(nn.Module):
    """Target-independent intent predictor for disjoint candidate groups."""

    def __init__(self, emb_dim: int, num_groups: int = 3, num_stats: int = 5,
                 hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.context_norm = nn.LayerNorm(emb_dim)
        self.net = nn.Sequential(
            nn.Linear(emb_dim + num_stats, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_groups),
        )
        self.register_buffer("stats_mean", torch.zeros(num_stats))
        self.register_buffer("stats_std", torch.ones(num_stats))

    def set_stats_normalization(self, stats: torch.Tensor) -> None:
        self.stats_mean.copy_(stats.mean(dim=0))
        self.stats_std.copy_(stats.std(dim=0, unbiased=False).clamp_min(1e-6))

    def forward(self, context: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        z = (stats - self.stats_mean) / self.stats_std
        return self.net(torch.cat([self.context_norm(context), z], dim=-1))


def multigroup_kl_shifts(target_mass: torch.Tensor, base_mass: torch.Tensor,
                         eps: float = 1e-12) -> torch.Tensor:
    """Closed-form KL projection shifts for disjoint, exhaustive groups.

    Adding the returned group-wise constants to logits produces group masses
    ``target_mass`` while preserving every within-group ranking. Empty groups
    must have zero target mass and receive a zero shift.
    """
    if target_mass.shape != base_mass.shape:
        raise ValueError("target_mass and base_mass must have the same shape")
    available = base_mass > eps
    invalid = (~available) & (target_mass > eps)
    if invalid.any():
        raise ValueError("positive target mass assigned to an empty group")
    shift = torch.zeros_like(base_mass)
    shift[available] = (target_mass[available].clamp_min(eps).log()
                        - base_mass[available].clamp_min(eps).log())
    return shift


def sequence_statistics(seq_items: torch.Tensor, valid_mask: torch.Tensor,
                        recent_window: int = 5) -> torch.Tensor:
    """Target-independent O(BL) sequence features used by the intent head."""
    # The sequences are left padded.  Sorting each row lets us count unique
    # positive IDs in O(B L log L) without a Python loop or a GPU-to-CPU sync.
    valid = valid_mask > 0
    lengths = valid.sum(dim=1).to(torch.float32)
    safe_lengths = lengths.clamp_min(1.0)
    values = torch.where(valid, seq_items, torch.zeros_like(seq_items))
    ordered = values.sort(dim=1).values
    is_positive = ordered > 0
    is_new = torch.ones_like(is_positive)
    is_new[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    unique = (is_positive & is_new).sum(dim=1).to(torch.float32)

    window = max(1, min(int(recent_window), seq_items.shape[1]))
    recent_values = values[:, -window:]
    recent_valid = valid[:, -window:]
    recent_lengths = recent_valid.sum(dim=1).to(torch.float32)
    recent_ordered = recent_values.sort(dim=1).values
    recent_positive = recent_ordered > 0
    recent_new = torch.ones_like(recent_positive)
    recent_new[:, 1:] = recent_ordered[:, 1:] != recent_ordered[:, :-1]
    recent_unique = (recent_positive & recent_new).sum(dim=1).to(torch.float32)

    # With left padding, the last valid item is the final column for every
    # non-empty sequence.
    last_items = values[:, -1]
    last_count = ((values == last_items[:, None]) & valid &
                  (last_items[:, None] > 0)).sum(dim=1).to(torch.float32)
    return torch.stack([
        torch.log1p(lengths),
        unique / safe_lengths,
        1.0 - unique / safe_lengths,
        1.0 - recent_unique / recent_lengths.clamp_min(1.0),
        last_count / safe_lengths,
    ], dim=1)
