import torch

from llm_icc.modeling.losses import sft_cross_entropy, weighted_distribution_mse


def test_masking_ignores_padded_options():
    pred = torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.2, 0.3, 0.4, 0.1]])
    target = torch.tensor([[0.5, 0.5, 9.0, 9.0], [0.2, 0.3, 0.4, 0.1]])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    assert weighted_distribution_mse(pred, target, mask, torch.ones(2)).item() == 0.0


def test_weighting_and_normalisation():
    pred = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[0.0, 1.0], [0.0, 1.0]])  # row 0 sse=2, row 1 sse=0
    mask = torch.ones(2, 2, dtype=torch.bool)
    w = torch.tensor([0.25, 0.75])
    assert abs(weighted_distribution_mse(pred, target, mask, w).item() - 0.5) < 1e-6


def test_float32_and_gradient():
    logits = torch.randn(3, 5, dtype=torch.bfloat16, requires_grad=True)
    p = torch.softmax(logits, -1)
    loss = weighted_distribution_mse(p, torch.full((3, 5), 0.2), torch.ones(3, 5, dtype=torch.bool), torch.ones(3))
    assert loss.dtype == torch.float32
    loss.backward()
    assert logits.grad is not None


def test_sft_ce_matches_manual():
    logits = torch.tensor([[2.0, 0.0, 1.0]])
    expected = -torch.log_softmax(logits, -1)[0, 2]
    assert torch.isclose(sft_cross_entropy(logits, torch.tensor([2])), expected)
