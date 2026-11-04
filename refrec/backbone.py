"""FARRec: Failure-Aware Routing Recommendation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertRouter(nn.Module):
    """Candidate-aware softmax router over four experts."""

    def __init__(
        self,
        emb_dim: int,
        router_hidden_dim: int,
        include_weak_score: bool = True,
    ) -> None:
        super().__init__()
        self.include_weak_score = include_weak_score
        # 6 vectors: h_recent, h_long, h_base, e_i, h_recent*e_i, h_long*e_i
        # stats: s_recent, s_long, [s_weak], s_repeat, item_pop
        stat_dim = 5 if include_weak_score else 4
        in_dim = emb_dim * 6 + stat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, 4),
        )

    def forward(
        self,
        h_recent: torch.Tensor,
        h_long: torch.Tensor,
        h_base: torch.Tensor,
        e_i: torch.Tensor,
        s_recent: torch.Tensor,
        s_long: torch.Tensor,
        s_weak: torch.Tensor,
        s_repeat: torch.Tensor,
        item_pop: torch.Tensor,
    ) -> torch.Tensor:
        stats = [s_recent, s_long, s_repeat, item_pop]
        if self.include_weak_score:
            stats.insert(2, s_weak)
        stat_tensor = torch.stack(stats, dim=-1)
        z = torch.cat([
            h_recent, h_long, h_base, e_i,
            h_recent * e_i, h_long * e_i,
            stat_tensor,
        ], dim=-1)
        return F.softmax(self.mlp(z), dim=-1)


class FARRec(nn.Module):
    """Failure-Aware Routing Recommendation with four evidence experts."""

    def __init__(
        self,
        num_items: int,
        num_categories: int,
        item2cat: dict[int, int],
        item_popularity: torch.Tensor | None = None,
        emb_dim: int = 128,
        cat_weight: float = 0.2,
        max_seq_len: int = 50,
        dropout: float = 0.1,
        gamma_weak: float = 0.8,
        gamma_repeat: float = 0.8,
        alpha_main: float = 1.0,
        alpha_recent: float = 1.0,
        cat_smooth_k: float = 2.0,
        router_hidden_dim: int = 64,
        router_fixed_weights: list[float] | None = None,
        disable_weak_expert: bool = False,
        disable_repeat_expert: bool = False,
        disable_router: bool = False,
        disable_recent_expert: bool = False,
        cosine_base: bool = False,
        use_gru: bool = False,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.num_categories = num_categories
        self.emb_dim = emb_dim
        self.cat_weight = cat_weight
        self.max_seq_len = max_seq_len
        self.gamma_weak = gamma_weak
        self.gamma_repeat = gamma_repeat
        self.alpha_main = alpha_main
        self.alpha_recent = alpha_recent
        self.cat_smooth_k = cat_smooth_k
        self.disable_weak_expert = disable_weak_expert
        self.disable_repeat_expert = disable_repeat_expert
        self.disable_router = disable_router
        self.disable_recent_expert = disable_recent_expert
        self.cosine_base = cosine_base
        self.use_gru = use_gru
        if use_gru:
            self.gru = nn.GRU(emb_dim, emb_dim, num_layers=1, batch_first=True)

        if router_fixed_weights is None:
            router_fixed_weights = [0.35, 0.35, 0.15, 0.15]
        self.register_buffer(
            "router_fixed_weights",
            torch.tensor(router_fixed_weights, dtype=torch.float32),
            persistent=False,
        )

        self.item_embedding = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories + 1, emb_dim, padding_idx=0)
        self.item_bias = nn.Embedding(num_items + 1, 1, padding_idx=0)
        nn.init.zeros_(self.item_bias.weight)

        cat_ids = torch.zeros(num_items + 1, dtype=torch.long)
        for iid, cid in item2cat.items():
            if 0 <= iid <= num_items:
                cat_ids[iid] = cid
        self.register_buffer("item2cat", cat_ids, persistent=False)

        if item_popularity is None:
            item_popularity = torch.zeros(num_items + 1, dtype=torch.float32)
        self.register_buffer("item_popularity", item_popularity, persistent=False)

        decay_weak = gamma_weak ** (max_seq_len - 1 - torch.arange(max_seq_len, dtype=torch.float32))
        decay_repeat = gamma_repeat ** (max_seq_len - 1 - torch.arange(max_seq_len, dtype=torch.float32))
        self.register_buffer("decay_weak_pos", decay_weak, persistent=False)
        self.register_buffer("decay_repeat_pos", decay_repeat, persistent=False)

        self.base_proj = nn.Linear(emb_dim * 3, emb_dim)
        self.base_ln = nn.LayerNorm(emb_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.router = ExpertRouter(
            emb_dim, router_hidden_dim,
            include_weak_score=not disable_weak_expert,
        )

    def item_repr(self, item_ids: torch.Tensor) -> torch.Tensor:
        return (
            self.item_embedding(item_ids)
            + self.cat_weight * self.category_embedding(self.item2cat[item_ids])
        )

    def seq_embedding(self, seq_items: torch.Tensor, seq_cats: torch.Tensor) -> torch.Tensor:
        emb = self.item_embedding(seq_items) + self.cat_weight * self.category_embedding(seq_cats)
        return self.emb_dropout(emb)

    def _readout(
        self,
        e_seq: torch.Tensor,
        valid_mask: torch.Tensor,
        recent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        h_long = (e_seq * valid_mask.unsqueeze(-1)).sum(dim=1) / denom

        if self.disable_recent_expert:
            h_recent = torch.zeros_like(h_long)
        else:
            recent_denom = recent_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            h_recent = (e_seq * recent_mask.unsqueeze(-1)).sum(dim=1) / recent_denom

        positions = torch.arange(e_seq.size(1), device=e_seq.device).unsqueeze(0)
        last_idx = (positions * valid_mask.long()).max(dim=1).values
        batch_idx = torch.arange(e_seq.size(0), device=e_seq.device)
        h_last = e_seq[batch_idx, last_idx]

        if self.use_gru:
            # Learned long-term readout: a GRU over the (left-padded, zero-masked)
            # item sequence, read at the last valid position, replaces naive
            # mean pooling as the long-interest expert.
            gru_out, _ = self.gru(e_seq)
            h_long = gru_out[batch_idx, last_idx]

        h_base = self.base_ln(self.base_proj(torch.cat([h_long, h_recent, h_last], dim=-1)))
        return h_recent, h_long, h_base, h_last

    def _compute_weakness(
        self,
        seq_cats: torch.Tensor,
        valid_mask: torch.Tensor,
        recent_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz = seq_cats.size(0)
        nc = self.num_categories
        device = seq_cats.device
        dtype = self.decay_weak_pos.dtype

        safe_cats = seq_cats.clamp(min=0, max=nc)
        valid = valid_mask > 0
        recent = recent_mask > 0

        long_count = torch.zeros(bsz, nc + 1, device=device, dtype=dtype)
        recent_count = torch.zeros(bsz, nc + 1, device=device, dtype=dtype)
        decayed_count = torch.zeros(bsz, nc + 1, device=device, dtype=dtype)

        decay_w = valid * self.decay_weak_pos.unsqueeze(0)
        long_count.scatter_add_(1, safe_cats, valid.float())
        recent_count.scatter_add_(1, safe_cats, recent.float())
        decayed_count.scatter_add_(1, safe_cats, decay_w)

        long_count = long_count[:, 1:]
        recent_count = recent_count[:, 1:]
        decayed_count = decayed_count[:, 1:]

        long_sum = long_count.sum(dim=1, keepdim=True).clamp_min(1e-8)
        recent_sum = recent_count.sum(dim=1, keepdim=True).clamp_min(1.0)
        p_long = long_count / long_sum
        p_recent = recent_count / recent_sum

        weakness = (
            torch.log1p(decayed_count)
            * (1.0 - p_long).clamp(min=0.0).pow(self.alpha_main)
            * (1.0 - p_recent).clamp(min=0.0).pow(self.alpha_recent)
        )
        shrinkage = long_count / (long_count + self.cat_smooth_k)
        return weakness * shrinkage

    def _gather_cat_weakness(self, weakness: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        cat_ids = self.item2cat[item_ids]
        safe = (cat_ids > 0) & (cat_ids <= self.num_categories)
        idx = (cat_ids - 1).clamp(min=0, max=self.num_categories - 1)
        gathered = weakness.gather(1, idx)
        return gathered * safe.float()

    def _repeat_scores(
        self,
        seq_items: torch.Tensor,
        valid_mask: torch.Tensor,
        cand_items: torch.Tensor,
    ) -> torch.Tensor:
        match = (
            cand_items.unsqueeze(-1) == seq_items.unsqueeze(1)
        ) & (valid_mask.unsqueeze(1) > 0) & (seq_items.unsqueeze(1) > 0)
        contrib = self.decay_repeat_pos.unsqueeze(0).unsqueeze(0)
        return (match.float() * contrib).sum(dim=-1)

    def _popularity(self, item_ids: torch.Tensor) -> torch.Tensor:
        safe = (item_ids > 0) & (item_ids <= self.num_items)
        pop = self.item_popularity[item_ids.clamp(min=0, max=self.num_items)]
        return pop * safe.float()

    def encode_sequence(
        self,
        seq_items: torch.Tensor,
        seq_cats: torch.Tensor,
        valid_mask: torch.Tensor,
        recent_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        e_seq = self.seq_embedding(seq_items, seq_cats)
        e_seq = e_seq * valid_mask.unsqueeze(-1)
        h_recent, h_long, h_base, _ = self._readout(e_seq, valid_mask, recent_mask)
        weakness = self._compute_weakness(seq_cats, valid_mask, recent_mask)
        return {
            "e_seq": e_seq,
            "h_recent": h_recent,
            "h_long": h_long,
            "h_base": h_base,
            "weakness": weakness,
        }

    def score_candidates(
        self,
        enc: dict[str, torch.Tensor],
        seq_items: torch.Tensor,
        valid_mask: torch.Tensor,
        cand_items: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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
        s_repeat = self._repeat_scores(seq_items, valid_mask, cand_items)
        item_pop = self._popularity(cand_items)

        if self.disable_recent_expert:
            s_recent = torch.zeros_like(s_recent)
        if self.disable_weak_expert:
            s_weak = torch.zeros_like(s_weak)
        if self.disable_repeat_expert:
            s_repeat = torch.zeros_like(s_repeat)

        if self.disable_router:
            w = self.router_fixed_weights.view(1, 1, 4)
            pi = w.expand(bsz, num_cand, -1)
        else:
            pi = self.router(
                h_recent_e, h_long_e, h_base_e, cand_emb,
                s_recent, s_long, s_weak, s_repeat, item_pop,
            )

        expert_scores = torch.stack([s_recent, s_long, s_weak, s_repeat], dim=-1)
        routed = (pi * expert_scores).sum(dim=-1)
        scores = routed + bias

        return {
            **enc,
            "scores": scores,
            "s_recent": s_recent,
            "s_long": s_long,
            "s_weak": s_weak,
            "s_repeat": s_repeat,
            "pi": pi,
            "weak_expert_scores": s_weak,
        }

    def encode(
        self,
        seq_items: torch.Tensor,
        seq_cats: torch.Tensor,
        valid_mask: torch.Tensor,
        weak_mask: torch.Tensor,
        recent_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del weak_mask
        return self.encode_sequence(seq_items, seq_cats, valid_mask, recent_mask)

    def forward(
        self,
        seq_items: torch.Tensor,
        seq_cats: torch.Tensor,
        valid_mask: torch.Tensor,
        weak_mask: torch.Tensor,
        recent_mask: torch.Tensor,
        sampled_items: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        enc = self.encode(seq_items, seq_cats, valid_mask, weak_mask, recent_mask)
        return self.score_candidates(enc, seq_items, valid_mask, sampled_items)

    def score_chunk(
        self,
        enc: dict[str, torch.Tensor],
        seq_items: torch.Tensor,
        valid_mask: torch.Tensor,
        chunk_items: torch.Tensor,
    ) -> torch.Tensor:
        bsz = enc["h_recent"].size(0)
        chunk_items_b = chunk_items.unsqueeze(0).expand(bsz, -1)
        out = self.score_candidates(enc, seq_items, valid_mask, chunk_items_b)
        return out["scores"]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def complexity_note(self, num_neg: int = 100) -> str:
        m = 1 + num_neg
        L, d, C = self.max_seq_len, self.emb_dim, self.num_categories
        train = f"Training: O(B*({L}d + {L}*{C} + {m}*d))"
        eval_note = f"Evaluation: O(B*({L}d + {L}*{C} + |I|*d)) chunked; no self-attention, no O(L^2)"
        return f"{train}; {eval_note}"
