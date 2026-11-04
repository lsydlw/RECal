"""Residual Evidence Fusion Recommendation (REFRec).

The model keeps FARRec's inexpensive sequence readout and item representation,
but replaces raw four-expert softmax fusion with a two-part score:

    base = mixture(recent, long)
    final = base + repeat_scale * (repeat - beta * base) + bias

The repeat residual is the portion of repeat evidence not linearly explained by
the base score.  No target, future interaction, or test-time label is used.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from refrec.backbone import FARRec


class REFRec(FARRec):
    """FARRec backbone with residual repeat-evidence fusion.

    ``refrec_orth_loss`` is returned on sampled training candidates.  It is the
    squared batch correlation of base and residual repeat scores and is only
    added to the objective when ``--lambda_refrec_orth`` is positive.
    """

    def __init__(
        self,
        *args,
        refrec_init_repeat_weight: float = 0.20,
        repeat_mode: str = "residual",
        fuse_mode: str = "residual",
        fix_beta: float | None = None,
        fix_scale: float | None = None,
        no_repeat_support: bool = False,
        **kwargs,
    ) -> None:
        if repeat_mode not in {"none", "raw", "residual"}:
            raise ValueError(f"Unknown repeat_mode: {repeat_mode}")
        if fuse_mode not in {"residual", "softmax", "residual_full"}:
            raise ValueError(f"Unknown fuse_mode: {fuse_mode}")
        self.repeat_mode = repeat_mode
        self.fuse_mode = fuse_mode
        self.fix_beta = fix_beta
        self.fix_scale = fix_scale
        self.no_repeat_support = no_repeat_support
        super().__init__(*args, **kwargs)
        # beta is user/sequence conditional but candidate-independent.
        # This keeps the additional computation O(Bd), not O(BMd).
        self.residual_projection = nn.Linear(self.emb_dim + 1, 1)
        # residual_full: per-expert projection coefficients (long, weak) for the
        # Gram-Schmidt base chain.  The repeat branch reuses residual_projection.
        self.base_residual_projection = nn.Linear(self.emb_dim + 1, 2)
        if refrec_init_repeat_weight <= 0:
            raise ValueError("refrec_init_repeat_weight must be positive")
        # Store the inverse-softplus value so the CLI argument is the actual
        # initial residual scale, rather than softplus(argument).
        inverse_softplus = math.log(math.expm1(float(refrec_init_repeat_weight)))
        self.repeat_scale_unconstrained = nn.Parameter(torch.tensor(inverse_softplus))
        self.long_scale_unconstrained = nn.Parameter(torch.tensor(inverse_softplus))
        self.weak_scale_unconstrained = nn.Parameter(torch.tensor(inverse_softplus))

    def encode_sequence(
        self,
        seq_items: torch.Tensor,
        seq_cats: torch.Tensor,
        valid_mask: torch.Tensor,
        recent_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return super().encode_sequence(seq_items, seq_cats, valid_mask, recent_mask)

    @staticmethod
    def _squared_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Differentiable squared Pearson correlation over sampled scores."""
        x = x.reshape(-1)
        y = y.reshape(-1)
        x = x - x.mean()
        y = y - y.mean()
        # Put epsilon inside sqrt. clamp_min after sqrt still leaves the
        # derivative of sqrt(0) singular and can produce NaN gradients even
        # when the external regularization weight is zero.
        x_scale = (x.square().mean() + 1e-8).sqrt()
        y_scale = (y.square().mean() + 1e-8).sqrt()
        return (x.mul(y).mean() / (x_scale * y_scale)).square()

    def score_candidates(
        self,
        enc: dict[str, torch.Tensor],
        seq_items: torch.Tensor,
        valid_mask: torch.Tensor,
        cand_items: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.fuse_mode == "softmax":
            # FARRec 原版：repeat 参与 4-expert softmax 路由（double-counting 对照）。
            out = super().score_candidates(enc, seq_items, valid_mask, cand_items)
            out["refrec_orth_loss"] = torch.zeros(
                (), device=out["scores"].device, dtype=out["scores"].dtype)
            return out
        h_recent = enc["h_recent"]
        h_long = enc["h_long"]
        h_base = enc["h_base"]
        cand_emb = self.item_repr(cand_items)
        bias = self.item_bias(cand_items).squeeze(-1)

        bsz, num_cand = cand_items.shape
        h_recent_e = h_recent.unsqueeze(1).expand(-1, num_cand, -1)
        h_long_e = h_long.unsqueeze(1).expand(-1, num_cand, -1)
        h_base_e = h_base.unsqueeze(1).expand(-1, num_cand, -1)
        if self.cosine_base:
            h_recent_e = F.normalize(h_recent_e, dim=-1)
            h_long_e = F.normalize(h_long_e, dim=-1)
            cand_emb = F.normalize(cand_emb, dim=-1)
        s_recent = torch.einsum("bcd,bcd->bc", h_recent_e, cand_emb)
        s_long = torch.einsum("bcd,bcd->bc", h_long_e, cand_emb)
        s_weak = self._gather_cat_weakness(enc["weakness"], cand_items)
        item_pop = self._popularity(cand_items)
        if self.disable_recent_expert:
            s_recent = torch.zeros_like(s_recent)
        if self.disable_weak_expert:
            s_weak = torch.zeros_like(s_weak)

        # Shared user/sequence context for the residual projections.
        seq_len = valid_mask.sum(dim=1, keepdim=True).float()
        context = torch.cat([h_base, torch.log1p(seq_len)], dim=-1)

        if self.fuse_mode == "residual_full":
            # Gram-Schmidt residual chain over the base experts:
            #   f0 = s_recent
            #   f1 = f0 + w_long * (s_long - b_long * f0)
            #   f2 = f1 + w_weak * (s_weak - b_weak * f1)
            # Each expert only contributes the part of itself not already
            # explained by everything fused so far (de-counts the A∩B overlap).
            b_long, b_weak = torch.sigmoid(self.base_residual_projection(context)).split(1, dim=-1)
            b_long = b_long.expand(-1, num_cand)
            b_weak = b_weak.expand(-1, num_cand)
            w_long = F.softplus(self.long_scale_unconstrained)
            w_weak = F.softplus(self.weak_scale_unconstrained)
            if self.disable_recent_expert:
                fused = s_long
            else:
                fused = s_recent
                fused = fused + w_long * (s_long - b_long * fused)
            if not self.disable_weak_expert:
                fused = fused + w_weak * (s_weak - b_weak * fused)
            base_score = fused
            # Diagnostic: [recent weight, long residual scale, weak residual scale].
            pi = torch.stack([
                torch.ones_like(s_recent),
                w_long * torch.ones_like(s_recent),
                w_weak * torch.ones_like(s_recent),
            ], dim=-1)
        else:
            # Build the strong candidate-aware base from recent/long/weak only.
            # A zero repeat input prevents the base router from seeing the evidence
            # that the residual branch is intended to test independently.
            zero_repeat = torch.zeros_like(s_recent)
            if self.disable_router:
                base_pi = self.router_fixed_weights[:3].view(1, 1, 3).expand(bsz, num_cand, -1)
            else:
                all_pi = self.router(
                    h_recent_e, h_long_e, h_base_e, cand_emb,
                    s_recent, s_long, s_weak, zero_repeat, item_pop,
                )
                base_pi = all_pi[..., :3]
            base_pi = base_pi / base_pi.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            base_experts = torch.stack([s_recent, s_long, s_weak], dim=-1)
            base_score = (base_pi * base_experts).sum(dim=-1)
            pi = torch.stack([base_pi[..., 0], base_pi[..., 1], base_pi[..., 2]], dim=-1)

        raw_repeat = self._repeat_scores(seq_items, valid_mask, cand_items)
        if self.disable_repeat_expert or self.repeat_mode == "none":
            raw_repeat = torch.zeros_like(raw_repeat)

        if self.fix_beta is not None:
            beta = torch.full((bsz, num_cand), float(self.fix_beta),
                              device=base_score.device, dtype=base_score.dtype)
        else:
            beta = torch.sigmoid(self.residual_projection(context)).expand(-1, num_cand)
        # A repeat expert has no evidence for an item absent from the history.
        # Without this mask, raw_repeat==0 would yield -beta*base_score for
        # every unseen item, unintentionally rescaling the entire catalog.
        if self.no_repeat_support:
            repeat_support = torch.ones_like(base_score)
        else:
            repeat_support = (raw_repeat > 0).to(base_score.dtype)
        if self.repeat_mode == "raw":
            repeat_residual = repeat_support * raw_repeat
        elif self.repeat_mode == "residual":
            repeat_residual = repeat_support * (raw_repeat - beta * base_score)
        else:
            repeat_residual = torch.zeros_like(base_score)
        if self.fix_scale is not None:
            repeat_scale = torch.full((bsz, num_cand), float(self.fix_scale),
                                      device=base_score.device, dtype=base_score.dtype)
        else:
            repeat_scale = F.softplus(self.repeat_scale_unconstrained)
        scores = base_score + repeat_scale * repeat_residual + bias

        orth_loss = self._squared_correlation(base_score, repeat_residual)

        return {
            **enc,
            "scores": scores,
            "s_recent": s_recent,
            "s_long": s_long,
            "s_weak": s_weak,
            "s_repeat": raw_repeat,
            "base_score": base_score,
            "repeat_residual": repeat_residual,
            "repeat_support": repeat_support,
            "repeat_beta": beta,
            "repeat_scale": repeat_scale.detach(),
            "pi": pi,
            "weak_expert_scores": s_weak,
            "refrec_orth_loss": orth_loss,
        }

    def complexity_note(self, num_neg: int = 100) -> str:
        m = 1 + num_neg
        return (
            f"Training: O(B*({self.max_seq_len}d + {m}*d)); "
            "Evaluation: O(B*(Ld + |I|*d)) chunked; residual fusion adds O(Bd), "
            "with no self-attention, O(L^2), or candidate-pair operations"
        )
