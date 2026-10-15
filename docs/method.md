# Method

Notation: N = 20 descriptors, n = options of an item, θ ~ N(μ, ς²) with μ = 0, ς = 1.

## Descriptors
Descriptor k covers (c_{k-1}, c_k], α_k = (c_k − μ)/ς.

- weight ω_k = Φ(α_k) − Φ(α_{k-1})
- midpoint θ_k = μ + ς [φ(α_{k-1}) − φ(α_k)] / ω_k  (mean of θ within the bin)

Implemented in `DescriptorScale.conditional_means / weights`, tested against numerical integration.

## NRM and eq. 4
P(y = v_i | θ) = softmax_i(a_i(θ − b_i)). With the correct option c and uniform distractors
(a_i = b_i = 0, i ≠ c): P(correct) = 1 / (1 + (n−1) e^{−a(θ−b_c)}). Choosing b_c = b − log(n−1)/a gives
P(correct) = σ(a(θ − b)) exactly, for any n. Distractors share the remaining mass equally.

## Training
1. **SFT (optional).** One answer sampled per (item, descriptor) from the target row, resampled each epoch;
   cross-entropy over the full vocabulary at the answer slot.
2. **Distribution correction.** p̂_ik = softmax over the n option-token logits at the answer slot.
   Loss = Σ_k ω_k Σ_i (p̂_ik − P_ik)² (normalised by Σω in a batch; computed in float32). Early stopping on
   the pooled ω-weighted dev loss; the step-0 (post-SFT) model counts as a candidate so the kept adapter is never
   worse than the start.

## Recovering difficulty
For each held-out item, the correct-option column of the predicted ICC is fitted with
b = argmin Σ_k ω_k (P_k − σ(θ_k − b))² (grid search + bounded refinement). `2PL` additionally fits a.
A line regressing reference `Difficulty` on predicted `b` over the dev items is applied to test predictions.
With `linear_recalibration: false`, the inverse of the standardize map is used instead.

## Errata in the preprint and how they are handled
- Table 2 "Critical" is (−∞, −3), not (∞, −3).
- Table 2 gap: Minimal ends at −1.8, Emerging starts at −1.5; default config extends Emerging down to −1.8.
  The validator rejects gaps/overlaps.
- Φ(x) = ∫_{−∞}^x, not ∫_0^x.
- Eq. 9 should read P(y = 1 | c_{k-1} < θ ≤ c_k).
- Eq. 12 and the vector after eq. 14: the intended vector is (P_j1 … P_jN) for the correct option.
- Appendix A asks for a free-text answer; token-probability training needs lettered options and a
  single option-token answer slot, so the prompt is adapted.
