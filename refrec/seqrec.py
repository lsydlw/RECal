"""Pluggable sequential-recommendation backbones + the residual repeat head.

The point of this module is that the *repeat de-double-counting head* is
backbone-agnostic.  Any sequential recommender produces a base score ``B`` for a
candidate item; we re-score it as

    score = B + w * support(R) * (R - beta * B)

where ``R`` is a generic, backbone-independent "did this item appear in the
history" signal (recency-decayed).  ``support(R)`` gates the residual so that an
item *absent* from the history is left untouched (``score = B``).

``repeat_head`` selects the ablation:
    * ``base``     -> score = B                       (plain backbone)
    * ``naive``    -> score = B + w * support(R) * R  (additive repeat, double-counts)
    * ``residual`` -> score = B + w * support(R) * (R - beta * B)   (the head)

The backbones only differ in how they turn a sequence into a single user
representation ``h_user``; everything downstream (item repr, bias, repeat head,
chunked full-catalog scoring) is shared.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# The pluggable repeat head
# --------------------------------------------------------------------------- #
class ResidualRepeatHead(nn.Module):
    """De-double-count the repeat evidence from a base score.

    ``base_score`` [B, C] and ``raw_repeat`` [B, C] are candidate-indexed
    tensors; ``context`` [B, D] is the user/sequence context used to predict the
    per-user projection coefficient ``beta``.
    """

    def __init__(
        self,
        emb_dim: int,
        mode: str = "residual",
        init_weight: float = 0.8,
        fix_beta: float | None = None,
        fix_scale: float | None = None,
        no_support: bool = False,
    ) -> None:
        super().__init__()
        if mode not in {"base", "naive", "residual"}:
            raise ValueError(f"Unknown repeat_head mode: {mode}")
        self.mode = mode
        self.fix_beta = fix_beta
        self.fix_scale = fix_scale
        self.no_support = no_support
        # beta is user/sequence conditional but candidate-independent -> O(Bd).
        self.residual_projection = nn.Linear(emb_dim + 1, 1)
        if init_weight <= 0:
            raise ValueError("init_weight must be positive")
        inv_softplus = math.log(math.expm1(float(init_weight)))
        self.repeat_scale_unconstrained = nn.Parameter(torch.tensor(inv_softplus))

    @staticmethod
    def _squared_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1) - x.reshape(-1).mean()
        y = y.reshape(-1) - y.reshape(-1).mean()
        xs = (x.square().mean() + 1e-8).sqrt()
        ys = (y.square().mean() + 1e-8).sqrt()
        return (x.mul(y).mean() / (xs * ys)).square()

    def forward(
        self,
        base_score: torch.Tensor,
        raw_repeat: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, _ = base_score.shape
        if self.fix_beta is not None:
            beta = torch.full((bsz, 1), float(self.fix_beta),
                              device=base_score.device, dtype=base_score.dtype)
        else:
            beta = torch.sigmoid(self.residual_projection(context))
        if self.fix_scale is not None:
            w = torch.full((), float(self.fix_scale),
                           device=base_score.device, dtype=base_score.dtype)
        else:
            w = F.softplus(self.repeat_scale_unconstrained)

        if self.no_support:
            support = torch.ones_like(base_score)
        else:
            # A repeat expert has no evidence for an item absent from the history.
            support = (raw_repeat > 0).to(base_score.dtype)

        if self.mode == "base":
            residual = torch.zeros_like(base_score)
        elif self.mode == "naive":
            residual = support * raw_repeat
        else:  # residual
            residual = support * (raw_repeat - beta * base_score)

        scores = base_score + w * residual
        orth_loss = self._squared_correlation(base_score, residual)
        return scores, orth_loss, beta, w.detach(), residual


# --------------------------------------------------------------------------- #
# Attention helpers
# --------------------------------------------------------------------------- #
class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]; attn_mask: [B, 1, L, L] additive (0 or -inf-ish)
        bsz, L, dim = x.shape
        h, hd = self.num_heads, self.head_dim
        q = self.Wq(x).view(bsz, L, h, hd).transpose(1, 2)
        k = self.Wk(x).view(bsz, L, h, hd).transpose(1, 2)
        v = self.Wv(x).view(bsz, L, h, hd).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, L, dim)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x


def _causal_padding_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    """Left-padded causal mask: position i attends to j<=i that are valid."""
    bsz, L = valid_mask.shape
    causal = torch.tril(torch.ones(L, L, device=valid_mask.device))
    pad = valid_mask.unsqueeze(-1).float()  # [B, L, 1]
    mask = (causal.unsqueeze(0) * pad).unsqueeze(1)  # [B, 1, L, L]
    return (1.0 - mask) * -1e9


def _bidir_padding_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    bsz, L = valid_mask.shape
    pad = valid_mask.unsqueeze(-1) * valid_mask.unsqueeze(-2)  # [B, L, L]
    return ((1.0 - pad.float()).unsqueeze(1)) * -1e9


# --------------------------------------------------------------------------- #
# Shared sequential-recommender base
# --------------------------------------------------------------------------- #
class SeqRecBase(nn.Module):
    """Everything shared across backbones except the sequence encoder.

    Subclasses implement :meth:`encode_sequence`, returning a dict with at least
    ``"h_user"`` [B, D].
    """

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
        gamma_repeat: float = 0.8,
        repeat_head: str = "residual",
        refrec_init_repeat_weight: float = 0.8,
        no_repeat_support: bool = False,
        fix_beta: float | None = None,
        fix_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.num_categories = num_categories
        self.emb_dim = emb_dim
        self.cat_weight = cat_weight
        self.max_seq_len = max_seq_len

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

        decay_repeat = gamma_repeat ** (max_seq_len - 1 - torch.arange(max_seq_len, dtype=torch.float32))
        self.register_buffer("decay_repeat_pos", decay_repeat, persistent=False)

        self.emb_dropout = nn.Dropout(dropout)
        self.head = ResidualRepeatHead(
            emb_dim, mode=repeat_head, init_weight=refrec_init_repeat_weight,
            fix_beta=fix_beta, fix_scale=fix_scale, no_support=no_repeat_support,
        )

    # -- item / sequence representation ---------------------------------- #
    def item_repr(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(item_ids) + self.cat_weight * self.category_embedding(self.item2cat[item_ids])

    def seq_embedding(self, seq_items: torch.Tensor, seq_cats: torch.Tensor) -> torch.Tensor:
        emb = self.item_embedding(seq_items) + self.cat_weight * self.category_embedding(seq_cats)
        return self.emb_dropout(emb)

    def _repeat_scores(self, seq_items: torch.Tensor, valid_mask: torch.Tensor, cand_items: torch.Tensor) -> torch.Tensor:
        match = (
            (cand_items.unsqueeze(-1) == seq_items.unsqueeze(1))
            & (valid_mask.unsqueeze(1) > 0)
            & (seq_items.unsqueeze(1) > 0)
        )
        contrib = self.decay_repeat_pos.unsqueeze(0).unsqueeze(0)
        return (match.float() * contrib).sum(dim=-1)

    def _last_hidden(self, h: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(h.size(1), device=h.device).unsqueeze(0)
        last_idx = (positions * valid_mask.long()).max(dim=1).values
        batch_idx = torch.arange(h.size(0), device=h.device)
        return h[batch_idx, last_idx]

    # -- per-backbone encoder -------------------------------------------- #
    def encode_sequence(self, seq_items, seq_cats, valid_mask, recent_mask):
        raise NotImplementedError

    # -- shared scoring / eval interface --------------------------------- #
    def encode(self, seq_items, seq_cats, valid_mask, weak_mask, recent_mask):
        del weak_mask
        return self.encode_sequence(seq_items, seq_cats, valid_mask, recent_mask)

    def compute_base_score(self, enc, seq_items, valid_mask, cand_items):
        h_user = enc["h_user"]
        cand_emb = self.item_repr(cand_items)
        bias = self.item_bias(cand_items).squeeze(-1)
        return (h_user.unsqueeze(1) * cand_emb).sum(dim=-1) + bias

    def score_candidates(self, enc, seq_items, valid_mask, cand_items):
        base_score = self.compute_base_score(enc, seq_items, valid_mask, cand_items)

        raw_repeat = self._repeat_scores(seq_items, valid_mask, cand_items)
        seq_len = valid_mask.sum(dim=1, keepdim=True).float()
        h_user = enc["h_user"]
        context = torch.cat([h_user, torch.log1p(seq_len)], dim=-1)
        scores, orth_loss, beta, w, residual = self.head(base_score, raw_repeat, context)

        return {
            **enc,
            "scores": scores,
            "base_score": base_score,
            "s_repeat": raw_repeat,
            "repeat_residual": residual,
            "repeat_beta": beta,
            "repeat_scale": w,
            "refrec_orth_loss": orth_loss,
        }

    def score_chunk(self, enc, seq_items, valid_mask, chunk_items):
        bsz = enc["h_user"].size(0)
        chunk_items_b = chunk_items.unsqueeze(0).expand(bsz, -1)
        return self.score_candidates(enc, seq_items, valid_mask, chunk_items_b)["scores"]

    def prepare_full_catalog(self, enc, seq_items, valid_mask, all_items, chunk_size=8192):
        """Attach batch-local constants needed for exact full-catalog scoring.

        Most dot-product backbones do not need a catalogue normalizer.  Models
        such as RepeatNet override this hook so chunked evaluation remains
        mathematically identical to scoring the whole catalogue at once.
        """
        return enc

    def forward(self, seq_items, seq_cats, valid_mask, weak_mask, recent_mask, sampled_items):
        enc = self.encode(seq_items, seq_cats, valid_mask, weak_mask, recent_mask)
        return self.score_candidates(enc, seq_items, valid_mask, sampled_items)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def complexity_note(self, num_neg: int = 100) -> str:
        m = 1 + num_neg
        return (f"Training: O(B*(Ld + m*d)); Evaluation: O(B*(Ld + |I|*d)) chunked; "
                f"repeat head adds O(Bd)")


# --------------------------------------------------------------------------- #
# Concrete backbones
# --------------------------------------------------------------------------- #
class GRU4Rec(SeqRecBase):
    def __init__(self, *args, gru_layers: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gru = nn.GRU(self.emb_dim, self.emb_dim, num_layers=gru_layers, batch_first=True)

    def encode_sequence(self, seq_items, seq_cats, valid_mask, recent_mask):
        e_seq = self.seq_embedding(seq_items, seq_cats) * valid_mask.unsqueeze(-1)
        gru_out, _ = self.gru(e_seq)
        return {"h_user": self._last_hidden(gru_out, valid_mask)}


class SASRec(SeqRecBase):
    def __init__(self, *args, num_blocks: int = 2, num_heads: int = 1, attn_dropout: float = 0.2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.emb_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(self.emb_dim, num_heads, attn_dropout) for _ in range(num_blocks)
        ])
        self.ln = nn.LayerNorm(self.emb_dim)

    def encode_sequence(self, seq_items, seq_cats, valid_mask, recent_mask):
        bsz, L = seq_items.shape
        e_seq = self.seq_embedding(seq_items, seq_cats)
        pos = torch.arange(L, device=seq_items.device).unsqueeze(0).expand(bsz, -1)
        x = e_seq + self.pos_emb(pos)
        attn_mask = _causal_padding_mask(valid_mask)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.ln(x) * valid_mask.unsqueeze(-1)
        return {"h_user": self._last_hidden(x, valid_mask)}


class BERT4Rec(SeqRecBase):
    """Bidirectional transformer; a [MASK] token is appended and its hidden state
    scores every candidate (standard next-item adaptation of BERT4Rec)."""

    def __init__(self, *args, num_blocks: int = 2, num_heads: int = 2, attn_dropout: float = 0.2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pos_emb = nn.Embedding(self.max_seq_len + 1, self.emb_dim)
        self.mask_emb = nn.Parameter(torch.randn(self.emb_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(self.emb_dim, num_heads, attn_dropout) for _ in range(num_blocks)
        ])
        self.ln = nn.LayerNorm(self.emb_dim)

    def encode_sequence(self, seq_items, seq_cats, valid_mask, recent_mask):
        bsz, L = seq_items.shape
        e_seq = self.seq_embedding(seq_items, seq_cats)
        mask_vec = self.mask_emb.view(1, 1, -1).expand(bsz, 1, -1)
        e_seq = torch.cat([e_seq, mask_vec], dim=1)  # [B, L+1, D]
        pos = torch.arange(L + 1, device=seq_items.device).unsqueeze(0).expand(bsz, -1)
        x = e_seq + self.pos_emb(pos)
        mask_valid = torch.cat([valid_mask, torch.ones(bsz, 1, device=valid_mask.device)], dim=1)
        attn_mask = _bidir_padding_mask(mask_valid)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.ln(x)
        return {"h_user": x[:, -1]}


class RepeatNet(SeqRecBase):
    """RepeatNet-style joint repeat/explore model (Ren et al., KDD 2019).

    A GRU encodes the session; two attention heads read it out: a *repeat*
    head (which past items to re-consume) and an *explore* head (a preference
    embedding for new items). A learned mode gate blends the two scores.
    """

    def __init__(self, *args, gru_layers: int = 1, attn_hidden: int = 64, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d = self.emb_dim
        self.gru = nn.GRU(d, d, num_layers=gru_layers, batch_first=True)
        self.w1 = nn.Linear(d, attn_hidden)
        self.w2 = nn.Linear(d, attn_hidden)
        self.v_r = nn.Linear(attn_hidden, 1)
        self.w3 = nn.Linear(d, attn_hidden)
        self.w4 = nn.Linear(d, attn_hidden)
        self.v_e = nn.Linear(attn_hidden, 1)
        self.mode_gate = nn.Linear(d, 1)

    def encode_sequence(self, seq_items, seq_cats, valid_mask, recent_mask):
        e_seq = self.seq_embedding(seq_items, seq_cats) * valid_mask.unsqueeze(-1)
        gru_out, _ = self.gru(e_seq)  # [B, L, D]
        h_last = self._last_hidden(gru_out, valid_mask).unsqueeze(1)  # [B, 1, D]

        neg = valid_mask.eq(0).float() * -1e9  # [B, L]

        a_r = torch.tanh(self.w1(h_last) + self.w2(gru_out))  # [B, L, H]
        e_r = self.v_r(a_r).squeeze(-1) + neg
        alpha_r = torch.softmax(e_r, dim=-1)  # [B, L]

        a_e = torch.tanh(self.w3(h_last) + self.w4(gru_out))
        e_e = self.v_e(a_e).squeeze(-1) + neg
        beta = torch.softmax(e_e, dim=-1)  # [B, L]

        h_explore = (beta.unsqueeze(-1) * e_seq).sum(dim=1)  # [B, D]
        p_r = torch.sigmoid(self.mode_gate(h_last.squeeze(1)))  # [B, 1]

        return {"h_user": h_explore, "alpha_r": alpha_r, "p_r": p_r}

    def _explore_logits(self, enc, cand_items):
        h_explore = enc["h_user"]
        cand_emb = self.item_repr(cand_items)
        bias = self.item_bias(cand_items).squeeze(-1)
        return (h_explore.unsqueeze(1) * cand_emb).sum(dim=-1) + bias

    @torch.no_grad()
    def prepare_full_catalog(self, enc, seq_items, valid_mask, all_items, chunk_size=8192):
        # RepeatNet mixes repeat and explore *probabilities*.  Compute the
        # explore partition function once per batch, then reuse it for every
        # catalogue chunk.  This prevents rankings from depending on chunk size.
        log_z = torch.full((enc["h_user"].size(0),), -torch.inf,
                           device=enc["h_user"].device)
        for start in range(0, len(all_items), chunk_size):
            ids = all_items[start:start + chunk_size]
            ids_b = ids.unsqueeze(0).expand(enc["h_user"].size(0), -1)
            logits = self._explore_logits(enc, ids_b)
            log_z = torch.logaddexp(log_z, torch.logsumexp(logits, dim=1))
        enc["repeatnet_explore_log_z"] = log_z.unsqueeze(1)
        return enc

    def compute_base_score(self, enc, seq_items, valid_mask, cand_items):
        alpha_r = enc["alpha_r"]  # [B, L]
        p_r = enc["p_r"].clamp(1e-7, 1.0 - 1e-7)  # [B, 1]

        explore_logits = self._explore_logits(enc, cand_items)
        if "repeatnet_explore_log_z" in enc:
            log_p_explore = explore_logits - enc["repeatnet_explore_log_z"]
        else:
            # Sampled-softmax training: normalize over the sampled candidate
            # set.  Full-catalog validation/test uses the exact normalizer above.
            log_p_explore = torch.log_softmax(explore_logits, dim=1)
        match = (
            (cand_items.unsqueeze(-1) == seq_items.unsqueeze(1))
            & (valid_mask.unsqueeze(1) > 0)
            & (seq_items.unsqueeze(1) > 0)
        )
        p_repeat_item = (match.float() * alpha_r.unsqueeze(1)).sum(dim=-1)
        log_p_repeat = torch.where(
            p_repeat_item > 0,
            torch.log(p_repeat_item.clamp_min(1e-30)),
            torch.full_like(p_repeat_item, -torch.inf),
        )
        return torch.logaddexp(
            torch.log(p_r) + log_p_repeat,
            torch.log1p(-p_r) + log_p_explore,
        )


BACKBONES = {
    "gru4rec": GRU4Rec,
    "sasrec": SASRec,
    "bert4rec": BERT4Rec,
    "repeatnet": RepeatNet,
}


def build_backbone(name: str, **kwargs) -> SeqRecBase:
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone: {name} (choose from {list(BACKBONES)})")
    return BACKBONES[name](**kwargs)
