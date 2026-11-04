"""Training objective used only by REFRec."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from refrec.utils import tqdm_bar


def train_one_epoch(model, loader, optimizer, device: torch.device, args: Any,
                    epoch: int, total_epochs: int, show_progress: bool = True) -> dict[str, float]:
    model.train()
    sums = {"loss": 0.0, "main_loss": 0.0, "orth_loss": 0.0}
    n_batches = 0
    pbar = tqdm_bar(loader, disable=not show_progress,
                    desc=f"Epoch {epoch}/{total_epochs} [train]", unit="batch", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["seq_items"], batch["seq_cats"], batch["valid_mask"],
                    batch["weak_mask"], batch["recent_mask"], batch["sampled_items"])
        main_loss = F.cross_entropy(out["scores"], batch["labels"])
        orth_loss = out["refrec_orth_loss"]
        # Do not attach a disabled auxiliary objective to the graph: 0 * NaN
        # is NaN in IEEE arithmetic and can corrupt otherwise valid gradients.
        if args.lambda_refrec_orth > 0:
            loss = main_loss + args.lambda_refrec_orth * orth_loss
        else:
            loss = main_loss
        # Optional backbone-native auxiliary objective.  FAESRec uses this
        # hook for its frequency-domain contrastive view and sparse mask
        # regularizer; existing backbones do not return the key and are
        # therefore bit-for-bit unchanged.
        auxiliary_loss = out.get("backbone_aux_loss")
        if auxiliary_loss is not None:
            loss = loss + auxiliary_loss
        loss.backward()
        optimizer.step()
        n_batches += 1
        sums["loss"] += float(loss.item())
        sums["main_loss"] += float(main_loss.detach().item())
        sums["orth_loss"] += float(orth_loss.detach().item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    if not n_batches:
        return {k: 0.0 for k in sums}
    return {"train_loss": sums["loss"] / n_batches,
            "main_loss": sums["main_loss"] / n_batches,
            "orth_loss": sums["orth_loss"] / n_batches}
