from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_distribution_mse(pred: torch.Tensor, target: torch.Tensor, option_mask: torch.Tensor,
                              weight: torch.Tensor) -> torch.Tensor:
    """sum_b w_b * sum_i mask_bi (pred_bi - target_bi)^2 / sum_b w_b, in float32."""
    pred, target, weight = pred.float(), target.float(), weight.float()
    sq = ((pred - target) ** 2) * option_mask.float()
    return (weight * sq.sum(dim=-1)).sum() / weight.sum()


def sft_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits (B, V) at the answer slot, labels (B,) token ids."""
    return F.cross_entropy(logits.float(), labels)
