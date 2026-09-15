# Marginalized retrieval loss: root cause and fix

## Observation

In `medrap-experiments` runs, marginalized-retrieval training loss was roughly
**an order of magnitude lower** than plain BCE loss.

## Setup

For each patient, the model retrieves `K` documents. Each document `k` has:

- a retrieval score `s_k = q · d_k` (dot product of query and document-key embeddings, dimension `D`)
- a prediction logit `l_k`

These combine into one probability via a softmax-weighted mixture:

```
P_ret(k | x) = softmax(s_1, ..., s_K)_k
P(y=1 | x)   = Σ_k  P_ret(k | x) · sigmoid(l_k)
loss         = -[ y·log P(y=1|x) + (1-y)·log P(y=0|x) ]
```

This is the intended design: a weighted average over documents, not a single-document prediction.

## Before: unscaled dot products saturate the softmax

`s_k` was computed as a **raw, unscaled dot product**. For embeddings with
roughly unit variance per dimension, a dot product of two `D`-dimensional
vectors has standard deviation `√D`. With Qwen3 embeddings, `D = 1024`, so
`√D ≈ 32` — score spreads of tens of units before training has learned
anything meaningful.

Feeding scores with that much spread into a softmax saturates it. Measured
directly (`K=8`, `D=1024`, random unit-variance embeddings):

|                        | mean top softmax weight |
| ---------------------- | ----------------------- |
| unscaled (before)      | **97.3%**               |
| uniform baseline (1/K) | 12.5%                   |

So `P_ret` was putting almost all its mass on one document — essentially at
random — regardless of true relevance. The mixture

```
P(y=1|x) = weighted average of sigmoid(l_1), ..., sigmoid(l_K)
```

collapsed in practice to

```
P(y=1|x) ≈ sigmoid(l_best)     (whichever document currently wins the softmax)
```

Because `s_k` is trained end-to-end, gradients could cheaply push `s_best`
higher for whichever document happened to fit that specific example's
label — a shortcut that lets training loss drop sharply without the model
learning transferable relevance.

*(Note: marginalized loss landing somewhat below plain BCE is expected on
its own — averaging probabilities before `-log` is provably ≤ the average
of the individual `-log` terms, by Jensen's inequality. That part isn't a
bug. The bug was the mixture collapsing to a single document instead of
actually averaging.)*

## After: scale by `1/√D`

Same fix used in Transformer self-attention (Vaswani et al., 2017), for the
same reason:

```
s_k = (q · d_k) / √D
```

Re-measured with scaling: mean top softmax weight drops from 97.3% → **35.9%**
— genuinely spread across documents instead of collapsed onto one.

`cosine` similarity was untouched: it's already bounded to `[-1, 1]` and has
no `D`-dependent variance blowup. Actual document retrieval (top-`K`
selection) is also unaffected — ranking order is invariant to a positive
scalar rescale; only the *differentiable re-scoring* used to weight the
marginalization depended on scale.

## Status

- Fix: `src/medrap/model/retrieval_scoring.py`, PR [#113](https://github.com/McDermottHealthAI/MedRAP/pull/113)
- Verified: the softmax-saturation numbers above, reproduced as a regression-test doctest
- Not yet verified: an actual before/after loss curve from a `medrap-experiments` re-run
