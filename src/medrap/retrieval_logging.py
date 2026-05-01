"""Lightweight batch-level diagnostics for Lightning/W&B logs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional
from torch import Tensor

from .types import ModelOutput, QueryOutput, RetrieverOutput

if TYPE_CHECKING:
    from meds_torchdata import MEDSTorchBatch


def count_unique_retrieved_documents(
    retrieval: RetrieverOutput, *, fingerprint_tokens: int = 32
) -> int | None:
    """Approximate number of distinct retrieved documents in this batch.

    Uses ``doc_ids`` when present; otherwise fingerprints by the first
    ``fingerprint_tokens`` token ids per document.

    Args:
        retrieval: Output of the retriever for one batch.
        fingerprint_tokens: Leading token ids used for uniqueness without
            document ids.

    Returns:
        Integer count, or ``None`` if neither ids nor usable tokens are present.

    Examples:
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
        ... )
        >>> count_unique_retrieved_documents(ro2, fingerprint_tokens=3)
        3
        >>> empty = RetrieverOutput(
        ...     doc_tokens=torch.empty(0, 1, 2, 3, dtype=torch.long),
        ...     doc_attention_mask=torch.empty(0, 1, 2, 3, dtype=torch.bool),
        ... )
        >>> count_unique_retrieved_documents(empty) is None
        True
    """
    if retrieval.doc_ids is not None:
        return int(torch.unique(retrieval.doc_ids.detach().long().flatten()).numel())

    doc_tokens = retrieval.doc_tokens
    if doc_tokens.numel() == 0:
        return None
    take = min(int(fingerprint_tokens), int(doc_tokens.shape[-1]))
    if take < 1:
        return None
    signatures = doc_tokens[..., :take].reshape(-1, take).detach().cpu()
    return int(torch.unique(signatures, dim=0).shape[0])


def _finite_tensor(x: Tensor) -> Tensor:
    x = x.detach().float()
    return x[torch.isfinite(x)]


def _std(x: Tensor) -> Tensor:
    return x.std(unbiased=False) if x.numel() > 1 else torch.zeros((), device=x.device)


def _probability_scalars(logits: Tensor, *, prefix: str) -> dict[str, Tensor]:
    logits = logits.detach().float()
    if logits.numel() == 0:
        return {}
    if logits.shape[-1] == 1:
        probs = torch.sigmoid(logits)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))) - (
            (1.0 - probs) * torch.log((1.0 - probs).clamp_min(1e-8))
        )
    else:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    return {
        f"{prefix}/prob_mean": probs.mean(),
        f"{prefix}/prob_std": _std(probs),
        f"{prefix}/entropy_mean": entropy.mean(),
    }


def _logit_scalars(logits: Tensor, *, prefix: str) -> dict[str, Tensor]:
    finite_logits = _finite_tensor(logits)
    if finite_logits.numel() == 0:
        return {f"{prefix}/finite_frac": torch.isfinite(logits.detach()).float().mean()}
    return {
        f"{prefix}/logits_mean": finite_logits.mean(),
        f"{prefix}/logits_std": _std(finite_logits),
        f"{prefix}/logits_max_abs": finite_logits.abs().max(),
        f"{prefix}/finite_frac": torch.isfinite(logits.detach()).float().mean(),
    }


def _query_scalars(query: QueryOutput, *, prefix: str) -> dict[str, Tensor]:
    queries = query.query_embeddings.detach().float()
    if queries.numel() == 0 or queries.ndim != 3:
        return {}
    flat = queries.reshape(-1, queries.shape[-1])
    out: dict[str, Tensor] = {
        f"{prefix}/norm_mean": flat.norm(dim=-1).mean(),
        f"{prefix}/dim_std_mean": flat.std(dim=0, unbiased=False).mean(),
    }
    if flat.shape[0] > 1:
        normalized = functional.normalize(flat, dim=-1)
        cosine = normalized @ normalized.T
        offdiag = cosine[~torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)]
        out[f"{prefix}/offdiag_cos_mean"] = offdiag.mean()
    return out


def _retrieval_id_scalars(retrieval: RetrieverOutput, *, prefix: str) -> dict[str, Tensor]:
    total = retrieval.doc_tokens.shape[0] * retrieval.doc_tokens.shape[1] * retrieval.doc_tokens.shape[2]
    unique_docs = count_unique_retrieved_documents(retrieval)
    out: dict[str, Tensor] = {}
    if unique_docs is not None and total > 0:
        out[f"{prefix}/unique_doc_ratio"] = torch.tensor(float(unique_docs) / float(total))
    if retrieval.doc_ids is None or retrieval.doc_ids.numel() == 0:
        return out
    top1 = retrieval.doc_ids[..., 0].detach().long().flatten()
    if top1.numel() == 0:
        return out
    unique_top1, counts = torch.unique(top1, return_counts=True)
    out[f"{prefix}/top1_unique_ratio"] = torch.tensor(
        float(unique_top1.numel()) / float(top1.numel()),
        device=top1.device,
    )
    out[f"{prefix}/top1_mode_frac"] = counts.max().float() / float(top1.numel())
    return out


def _retrieval_score_scalars(scores: Tensor, *, prefix: str) -> dict[str, Tensor]:
    scores = scores.detach().float()
    if scores.numel() == 0 or scores.shape[-1] < 1:
        return {}
    flat = scores.reshape(-1, scores.shape[-1])
    return {
        f"{prefix}/score_mean": flat.mean(),
        f"{prefix}/score_std": _std(flat),
        f"{prefix}/top1_score_mean": flat[:, 0].mean(),
    }


def _differentiable_score_scalars(scores: Tensor, *, prefix: str) -> dict[str, Tensor]:
    scores = scores.detach().float()
    if scores.numel() == 0 or scores.shape[-1] < 1:
        return {}
    flat = scores.reshape(-1, scores.shape[-1])
    probs = torch.softmax(flat, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    out = {
        f"{prefix}/score_mean": flat.mean(),
        f"{prefix}/score_std": _std(flat),
        f"{prefix}/score_entropy_mean": entropy.mean(),
        f"{prefix}/effective_k_mean": torch.exp(entropy).mean(),
    }
    if flat.shape[-1] > 1:
        sorted_scores = torch.sort(flat, dim=-1, descending=True).values
        out[f"{prefix}/top1_top2_margin_mean"] = (sorted_scores[:, 0] - sorted_scores[:, 1]).mean()
        out[f"{prefix}/top1_topk_margin_mean"] = (sorted_scores[:, 0] - flat.mean(dim=-1)).mean()
    return out


def _validation_diagnostic_logs(logs: dict[str, Tensor]) -> dict[str, Tensor]:
    """Keep validation diagnostics compact enough for routine W&B runs."""
    keep_suffixes = (
        "/finite_frac",
        "/logits_std",
        "/logits_max_abs",
        "/prob_mean",
        "/prob_std",
        "/entropy_mean",
        "/pad_fraction",
        "/valid_tokens_mean",
        "/norm_mean",
        "/dim_std_mean",
        "/offdiag_cos_mean",
        "/unique_doc_ratio",
        "/top1_unique_ratio",
        "/top1_mode_frac",
        "/differentiable/score_entropy_mean",
        "/differentiable/effective_k_mean",
        "/differentiable/top1_top2_margin_mean",
    )
    return {name: value for name, value in logs.items() if name.endswith(keep_suffixes)}


def _diagnostic_prefix(component: str, *, stage: str) -> str:
    if stage == "val":
        return f"val_diagnostics/{component}"
    return f"{component}/{stage}"


def _mask_scalars(batch: MEDSTorchBatch, *, prefix: str) -> dict[str, Tensor]:
    code = getattr(batch, "code", None)
    if not isinstance(code, Tensor) or code.numel() == 0:
        return {}
    valid = code != 0
    return {
        f"{prefix}/pad_fraction": (~valid).float().mean(),
        f"{prefix}/valid_tokens_mean": valid.sum(dim=-1).float().mean(),
    }


def model_diagnostic_scalars(
    predictions: ModelOutput,
    batch: MEDSTorchBatch,
    *,
    stage: str,
) -> dict[str, Tensor]:
    """Return a small, curated set of scalar diagnostics.

    Training metrics are grouped by W&B top-level section:
    ``prediction/train``, ``query/train``, ``retrieval/train``, and
    ``mask/train``. Validation diagnostics are grouped separately under
    ``val_diagnostics/*`` so they do not pollute the train diagnostic panels.

    Args:
        predictions: Model output from a supervised step.
        batch: Raw batch used for the forward pass.
        stage: Logging stage such as ``"train"`` or ``"val"``.

    Returns:
        Mapping from metric name to detached scalar tensor.

    Examples:
        >>> query = QueryOutput(torch.eye(2).reshape(2, 1, 2))
        >>> retriever = RetrieverOutput(
        ...     doc_tokens=torch.ones(2, 1, 2, 3, dtype=torch.long),
        ...     doc_attention_mask=torch.ones(2, 1, 2, 3, dtype=torch.bool),
        ...     doc_ids=torch.tensor([[[1, 2]], [[2, 3]]]),
        ...     doc_scores=torch.tensor([[[2.0, 1.0]], [[0.5, 0.25]]]),
        ... )
        >>> batch = MEDSTorchBatch(
        ...     code=torch.tensor([[1, 0], [2, 3]]),
        ...     numeric_value=torch.zeros(2, 2),
        ...     numeric_value_mask=torch.zeros(2, 2, dtype=torch.bool),
        ...     time_delta_days=torch.zeros(2, 2),
        ... )
        >>> out = ModelOutput(
        ...     logits=torch.tensor([[0.0], [1.0]]),
        ...     metadata={"query_output": query, "retriever_output": retriever},
        ... )
        >>> logs = model_diagnostic_scalars(out, batch, stage="train")
        >>> sorted(k for k in logs if k.startswith("retrieval/train/"))[:2]
        ['retrieval/train/score_mean', 'retrieval/train/score_std']
        >>> "mask/train/pad_fraction" in logs and "query/train/norm_mean" in logs
        True
        >>> val_logs = model_diagnostic_scalars(out, batch, stage="val")
        >>> "val_diagnostics/retrieval/unique_doc_ratio" in val_logs
        True
        >>> "val_diagnostics/retrieval/score_mean" in val_logs
        False
    """
    logs: dict[str, Tensor] = {}
    prediction_prefix = _diagnostic_prefix("prediction", stage=stage)
    retrieval_prefix = _diagnostic_prefix("retrieval", stage=stage)
    logs.update(_logit_scalars(predictions.logits, prefix=prediction_prefix))
    logs.update(_probability_scalars(predictions.logits, prefix=prediction_prefix))
    logs.update(_mask_scalars(batch, prefix=_diagnostic_prefix("mask", stage=stage)))

    query = predictions.metadata.get("query_output")
    if isinstance(query, QueryOutput):
        logs.update(_query_scalars(query, prefix=_diagnostic_prefix("query", stage=stage)))

    retrieval = predictions.metadata.get("retriever_output")
    if isinstance(retrieval, RetrieverOutput):
        logs.update(_retrieval_id_scalars(retrieval, prefix=retrieval_prefix))
        if retrieval.doc_scores is not None:
            logs.update(_retrieval_score_scalars(retrieval.doc_scores, prefix=retrieval_prefix))

    differentiable_scores = predictions.metadata.get("differentiable_doc_scores")
    if isinstance(differentiable_scores, Tensor):
        logs.update(
            _differentiable_score_scalars(
                differentiable_scores,
                prefix=f"{retrieval_prefix}/differentiable",
            )
        )

    if stage == "val":
        logs = _validation_diagnostic_logs(logs)
    return {name: value.detach() for name, value in logs.items()}
