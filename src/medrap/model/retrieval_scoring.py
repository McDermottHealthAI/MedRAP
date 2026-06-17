"""Differentiable scores between queries and retrieved document keys."""

from torch import Tensor
from torch.nn import functional as nn_functional


def differentiable_retrieval_scores(
    query_embeddings: Tensor,
    doc_key_embeddings: Tensor,
    *,
    similarity: str = "dot",
) -> Tensor:
    """Compute per-query, per-document scores with gradients to ``query_embeddings``.

    Args:
        query_embeddings: Tensor shaped ``(B, R, D)``.
        doc_key_embeddings: Tensor shaped ``(B, R, K, D)``.
        similarity: ``\"dot\"`` or ``\"cosine\"`` (same convention as
            :class:`medrap.retrievers.InMemoryRetriever`).

    Returns:
        Tensor shaped ``(B, K)`` when ``R == 1``, else ``(B, R, K)``.

    Raises:
        ValueError: If shapes are incompatible, ``R`` does not match, or
            ``similarity`` is unknown.

    Examples:
        >>> import torch
        >>> q = torch.randn(2, 1, 4, requires_grad=True)
        >>> k = torch.randn(2, 1, 3, 4)
        >>> differentiable_retrieval_scores(q, k).shape
        torch.Size([2, 3])
        >>> differentiable_retrieval_scores(q, k, similarity="cosine").shape
        torch.Size([2, 3])
        >>> differentiable_retrieval_scores(torch.randn(2, 2, 4), torch.randn(2, 2, 3, 4)).shape
        torch.Size([2, 2, 3])
        >>> differentiable_retrieval_scores(torch.zeros(2, 4), torch.zeros(2, 1, 3, 4))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: query_embeddings must be (B, R, D)...
        >>> differentiable_retrieval_scores(torch.zeros(2, 1, 4), torch.zeros(2, 1, 3))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: doc_key_embeddings must be (B, R, K, D)...
        >>> differentiable_retrieval_scores(
        ...     torch.zeros(2, 1, 4), torch.zeros(2, 1, 3, 5)
        ... )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: ...align on (B, R, D)...
        >>> differentiable_retrieval_scores(
        ...     torch.zeros(2, 1, 4), torch.zeros(3, 1, 3, 4)
        ... )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: ...align on (B, R, D)...
        >>> differentiable_retrieval_scores(
        ...     torch.zeros(2, 1, 4), torch.zeros(2, 1, 3, 4), similarity="l2"
        ... )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: similarity must be 'dot' or 'cosine'...
    """
    if query_embeddings.ndim != 3:
        raise ValueError(f"query_embeddings must be (B, R, D), got {tuple(query_embeddings.shape)}")
    if doc_key_embeddings.ndim != 4:
        raise ValueError(f"doc_key_embeddings must be (B, R, K, D), got {tuple(doc_key_embeddings.shape)}")
    q = query_embeddings.float()
    k = doc_key_embeddings.float()
    if q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError(
            "query_embeddings and doc_key_embeddings must align on (B, R, D); "
            f"got query={tuple(q.shape)}, keys={tuple(k.shape)}"
        )
    if similarity == "cosine":
        qn = nn_functional.normalize(q, dim=-1)
        kn = nn_functional.normalize(k, dim=-1)
        scores = (qn.unsqueeze(2) * kn).sum(dim=-1)
    elif similarity == "dot":
        scores = (q.unsqueeze(2) * k).sum(dim=-1)
    else:
        raise ValueError(f"similarity must be 'dot' or 'cosine', got {similarity!r}")
    if scores.shape[1] == 1:
        return scores.squeeze(1)
    return scores
