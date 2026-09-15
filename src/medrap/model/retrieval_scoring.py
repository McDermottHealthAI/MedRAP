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

    ``similarity=\"dot\"`` scales the raw dot product by ``1 / sqrt(D)`` -- the
    same scaling self-attention uses (Vaswani et al. 2017) -- because these
    scores feed directly into a ``softmax`` for marginalizing predictions
    over retrieved documents (see ``MultiTaskBCEMarginalizedLoss``,
    ``MarginalizedRetrievalLoss``). For embeddings with roughly unit
    per-dimension variance, an unscaled dot product has standard deviation
    ``sqrt(D)``, which saturates ``softmax`` to a near one-hot distribution
    over ``K`` documents regardless of how semantically close the query
    actually is to each key -- collapsing the marginalization onto a single
    (effectively arbitrary, pre-training) document and letting the retriever
    cheaply minimize training loss by memorizing per-example document
    preference rather than learning real relevance.

    Args:
        query_embeddings: Tensor shaped ``(B, R, D)``.
        doc_key_embeddings: Tensor shaped ``(B, R, K, D)``.
        similarity: ``\"dot\"`` or ``\"cosine\"`` (same convention as
            :class:`medrap.retrievers.InMemoryRetriever`). Unlike
            ``InMemoryRetriever``, which uses these only to rank documents (a
            transform invariant to scale), the scores computed here are also
            used as softmax logits, where scale is not invariant.

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

        ``dot`` similarity divides by ``sqrt(D)`` so score variance does not grow
        with embedding dimension -- unscaled, high-dimensional embeddings with
        unit per-dimension variance would saturate the downstream softmax:

        >>> _ = torch.manual_seed(0)
        >>> D = 1024
        >>> q_big = torch.randn(4000, 1, D)
        >>> k_big = torch.randn(4000, 1, 8, D)
        >>> scores = differentiable_retrieval_scores(q_big, k_big)
        >>> round(scores.std().item())
        1
        >>> bool(torch.softmax(scores, dim=-1).max(dim=-1).values.mean() < 0.5)
        True
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
        scores = (q.unsqueeze(2) * k).sum(dim=-1) / (q.shape[-1] ** 0.5)
    else:
        raise ValueError(f"similarity must be 'dot' or 'cosine', got {similarity!r}")
    if scores.shape[1] == 1:
        return scores.squeeze(1)
    return scores
