"""Batch-level retrieval diagnostics for training logs (WandB / Lightning)."""

from __future__ import annotations

import torch
from torch import Tensor

from .types import ModelOutput, RetrieverOutput


def count_unique_retrieved_documents(
    retrieval: RetrieverOutput, *, fingerprint_tokens: int = 32
) -> int | None:
    """Approximate number of distinct retrieved documents in this batch.

    Uses ``doc_ids`` when present; otherwise fingerprints by the first ``fingerprint_tokens``
    token ids per document (handles ``doc_ids_column=null`` corpora).

    Args:
        retrieval: Output of the retriever for one batch.
        fingerprint_tokens: Truncate token sequences to this many leading ids for uniqueness.

    Returns:
        An integer count, or ``None`` if neither ids nor tokens are available.

    Examples:
        >>> import torch
        >>> ro = RetrieverOutput(
        ...     doc_tokens=torch.zeros(2, 1, 2, 3, dtype=torch.long),
        ...     doc_attention_mask=torch.ones(2, 1, 2, 3, dtype=torch.bool),
        ...     doc_ids=torch.tensor([[[1, 2]], [[2, 1]]]),
        ... )
        >>> count_unique_retrieved_documents(ro)
        2
        >>> ro2 = RetrieverOutput(
        ...     doc_tokens=torch.tensor([[[[1, 2, 3], [1, 2, 3]]], [[[4, 5, 6], [1, 2, 0]]]]),
        ...     doc_attention_mask=torch.ones(2, 1, 2, 3, dtype=torch.bool),
        ...     doc_ids=None,
        ... )
        >>> count_unique_retrieved_documents(ro2, fingerprint_tokens=3)
        3
    """
    if retrieval.doc_ids is not None:
        return int(torch.unique(retrieval.doc_ids.detach().long().flatten()).numel())

    dt = retrieval.doc_tokens
    if dt is None or dt.numel() == 0:
        return None
    take = min(int(fingerprint_tokens), int(dt.shape[-1]))
    if take < 1:
        return None
    # (B, R, K, S) -> (B*R*K, take)
    sig = dt[..., :take].reshape(-1, take).detach().cpu()
    return int(torch.unique(sig, dim=0).shape[0])


def retrieval_diagnostic_scalars(predictions: ModelOutput, targets: Tensor) -> dict[str, float]:
    """Scalar stats for logging from one forward pass (marginalized / differentiable-scores path).

    Expects ``predictions.metadata`` to contain ``differentiable_doc_scores`` ``(B, K)`` and
    optionally ``retriever_output`` for uniqueness. Targets must be float/bool 0/1 of shape ``(B,)``.

    Returns:
        Flat dict of metric name (without ``train/`` prefix) to float. Empty if scores missing.

    Examples:
        >>> import torch
        >>> from medrap.types import ModelOutput, RetrieverOutput
        >>> pred = ModelOutput(
        ...     logits=torch.zeros(2, 2),
        ...     metadata={
        ...         "differentiable_doc_scores": torch.tensor([[1.0, 2.0], [0.5, 0.25]]),
        ...         "retriever_output": RetrieverOutput(
        ...             doc_tokens=torch.zeros(2, 1, 2, 4, dtype=torch.long),
        ...             doc_attention_mask=torch.ones(2, 1, 2, 4, dtype=torch.bool),
        ...             doc_ids=torch.tensor([[[0, 1]], [[0, 1]]]),
        ...         ),
        ...     },
        ... )
        >>> d = retrieval_diagnostic_scalars(pred, torch.tensor([1.0, 0.0]))
        >>> "retrieval/doc_score_mean" in d and d["retrieval/unique_docs_per_batch"] == 2.0
        True
        >>> d["retrieval/doc_score_positive_vs_negative"] > 0
        True
        >>> retrieval_diagnostic_scalars(ModelOutput(logits=torch.zeros(2, 1)), torch.zeros(2))
        {}
    """
    meta = predictions.metadata
    scores = meta.get("differentiable_doc_scores")
    if not isinstance(scores, Tensor):
        return {}

    s = scores.detach().float()
    if s.ndim != 2:
        return {}

    out: dict[str, float] = {
        "retrieval/doc_score_mean": float(s.mean().item()),
        "retrieval/doc_score_std": float(s.std(unbiased=False).item()),
    }

    row_mean = s.mean(dim=-1)
    t = targets.detach().float().reshape(-1)
    if row_mean.shape[0] != t.shape[0]:
        return out

    pos = t > 0.5
    neg = ~pos
    if pos.any():
        out["retrieval/doc_score_mean_positive"] = float(row_mean[pos].mean().item())
    if neg.any():
        out["retrieval/doc_score_mean_negative"] = float(row_mean[neg].mean().item())
    if pos.any() and neg.any():
        out["retrieval/doc_score_positive_vs_negative"] = float(
            row_mean[pos].mean().item() - row_mean[neg].mean().item()
        )

    ro = meta.get("retriever_output")
    if isinstance(ro, RetrieverOutput):
        nuniq = count_unique_retrieved_documents(ro)
        if nuniq is not None:
            out["retrieval/unique_docs_per_batch"] = float(nuniq)

    return out
