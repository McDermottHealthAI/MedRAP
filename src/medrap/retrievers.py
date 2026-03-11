"""Retrieval backends for RAP pipeline composition."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from .types import RetrieverOutput


class Retriever(nn.Module, ABC):
    """Abstract base class for retrievers.

    Subclasses must implement :meth:`retrieve`. The ``forward`` method
    delegates to ``retrieve`` so retrievers can be used as standard
    ``nn.Module`` objects.
    """

    @abstractmethod
    def retrieve(self, query_embeddings: Tensor) -> RetrieverOutput:
        """Retrieve documents for the given query embeddings.

        Args:
            query_embeddings: Query tensor with shape ``(B, R, D_ret)``.

        Returns:
            A ``RetrieverOutput``.
        """

    def forward(self, query_embeddings: Tensor) -> RetrieverOutput:
        """Call ``retrieve``."""
        return self.retrieve(query_embeddings)


class InMemoryRetriever(Retriever):
    """In-memory top-k retriever over fixed document keys.

    Args:
        doc_key_embeddings: Document key matrix with shape ``(N_docs, D_ret)``.
        doc_tokens: Document token ids with shape ``(N_docs, S_doc)``.
        doc_attention_mask: Document mask matching ``doc_tokens``.
        k: Number of documents to return per query.
        similarity: Similarity function used for retrieval. Supported values are
            ``"dot"`` and ``"cosine"``.
        doc_ids: Optional integer document ids with shape ``(N_docs,)``.
    """

    def __init__(
        self,
        *,
        doc_key_embeddings: Tensor,
        doc_tokens: Tensor,
        doc_attention_mask: Tensor,
        k: int = 1,
        similarity: str = "dot",
        doc_ids: Tensor | None = None,
    ) -> None:
        super().__init__()
        if doc_key_embeddings.ndim != 2:
            raise ValueError("doc_key_embeddings must have shape (N_docs, D_ret)")
        if doc_tokens.ndim != 2:
            raise ValueError("doc_tokens must have shape (N_docs, S_doc)")
        if doc_attention_mask.shape != doc_tokens.shape:
            raise ValueError("doc_attention_mask must match doc_tokens shape")
        if doc_key_embeddings.shape[0] != doc_tokens.shape[0]:
            raise ValueError("doc_key_embeddings and doc_tokens must have the same number of documents")
        if doc_ids is not None and doc_ids.shape != (doc_tokens.shape[0],):
            raise ValueError("doc_ids must have shape (N_docs,)")
        if similarity not in {"dot", "cosine"}:
            raise ValueError("similarity must be either 'dot' or 'cosine'")
        if k < 1 or k > doc_tokens.shape[0]:
            raise ValueError("k must be between 1 and the number of documents")

        self.k = k
        self.similarity = similarity
        self.register_buffer("_doc_key_embeddings", doc_key_embeddings.to(torch.float32))
        self.register_buffer("_doc_tokens", doc_tokens.to(torch.long))
        self.register_buffer("_doc_attention_mask", doc_attention_mask.to(torch.bool))
        if doc_ids is None:
            doc_ids = torch.arange(doc_tokens.shape[0], dtype=torch.long)
        self.register_buffer("_doc_ids", doc_ids.to(torch.long))

    def retrieve(self, query_embeddings: Tensor) -> RetrieverOutput:
        """Retrieve top-k documents for each query.

        Args:
            query_embeddings: Query tensor with shape ``(B, R, D_ret)`` on any
            device.

        Returns:
            ``RetrieverOutput`` on same device as ``query_embeddings`` with:
                - ``doc_tokens`` shaped ``(B, R, K, S_doc)``
                - ``doc_attention_mask`` shaped ``(B, R, K, S_doc)``
                - ``doc_scores`` shaped ``(B, R, K)``
                - ``doc_ids`` shaped ``(B, R, K)``
                - ``doc_key_embeddings`` shaped ``(B, R, K, D_ret)``

        Examples:
            >>> retriever = InMemoryRetriever(
            ...     doc_key_embeddings=torch.FloatTensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            ...     doc_tokens=torch.LongTensor([[10, 11], [20, 21], [30, 31]]),
            ...     doc_attention_mask=torch.BoolTensor([[True, True], [True, True], [True, False]]),
            ...     k=2,
            ... )
            >>> out = retriever.retrieve(torch.FloatTensor([[[1.0, -0.2]]]))
            >>> out.doc_ids.tolist()
            [[[0, 2]]]
            >>> tuple(out.doc_tokens.shape)
            (1, 1, 2, 2)
            >>> tuple(out.doc_key_embeddings.shape)
            (1, 1, 2, 2)
        """
        if query_embeddings.ndim != 3:
            raise ValueError("query_embeddings must have shape (B, R, D_ret)")
        doc_key_embeddings = cast("Tensor", self._doc_key_embeddings)
        doc_tokens = cast("Tensor", self._doc_tokens)
        doc_attention_mask = cast("Tensor", self._doc_attention_mask)
        doc_ids = cast("Tensor", self._doc_ids)

        if query_embeddings.shape[-1] != doc_key_embeddings.shape[-1]:
            raise ValueError("query_embeddings last dimension must match document key dimension")

        if self.similarity == "cosine":
            query_vectors = torch.nn.functional.normalize(query_embeddings.to(torch.float32), dim=-1)
            doc_keys = torch.nn.functional.normalize(doc_key_embeddings, dim=-1)
        else:
            query_vectors = query_embeddings.to(torch.float32)
            doc_keys = doc_key_embeddings

        scores = torch.einsum("brd,nd->brn", query_vectors, doc_keys)
        top_scores, top_indices = torch.topk(scores, k=self.k, dim=-1)

        retrieved_doc_tokens = doc_tokens[top_indices]
        retrieved_doc_attention_mask = doc_attention_mask[top_indices]
        retrieved_doc_ids = doc_ids[top_indices]
        retrieved_doc_key_embeddings = doc_key_embeddings[top_indices]

        return RetrieverOutput(
            doc_tokens=retrieved_doc_tokens,
            doc_attention_mask=retrieved_doc_attention_mask,
            doc_scores=top_scores,
            doc_ids=retrieved_doc_ids,
            doc_key_embeddings=retrieved_doc_key_embeddings,
        )


def load_in_memory_retriever_from_pt(
    *,
    bundle_path: str | Path,
    k: int = 1,
    similarity: str = "dot",
) -> InMemoryRetriever:
    """Load an ``InMemoryRetriever`` from a serialized ``.pt`` bundle.

    Args:
        bundle_path: Path to a ``torch.save``-produced bundle containing
            ``doc_key_embeddings``, ``doc_tokens``, and ``doc_attention_mask``.
            The bundle may optionally include ``doc_ids``.
        k: Number of documents to return per query.
        similarity: Similarity function used for retrieval. Supported values are
            ``"dot"`` and ``"cosine"``.

    Returns:
        A configured ``InMemoryRetriever``.

    Examples:
        >>> from pathlib import Path
        >>> import torch
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     bundle_path = Path(tmp_dir) / "retriever.pt"
        ...     torch.save(
        ...         {
        ...             "doc_key_embeddings": torch.FloatTensor([[1.0, 0.0], [0.0, 1.0]]),
        ...             "doc_tokens": torch.LongTensor([[10, 11], [20, 21]]),
        ...             "doc_attention_mask": torch.BoolTensor([[True, True], [True, False]]),
        ...             "doc_ids": torch.LongTensor([7, 8]),
        ...         },
        ...         bundle_path,
        ...     )
        ...     retriever = load_in_memory_retriever_from_pt(bundle_path=bundle_path, k=1)
        >>> tuple(retriever.retrieve(torch.FloatTensor([[[1.0, 0.0]]])).doc_ids.shape)
        (1, 1, 1)
    """
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    return InMemoryRetriever(
        doc_key_embeddings=torch.as_tensor(bundle["doc_key_embeddings"], dtype=torch.float32),
        doc_tokens=torch.as_tensor(bundle["doc_tokens"], dtype=torch.long),
        doc_attention_mask=torch.as_tensor(bundle["doc_attention_mask"], dtype=torch.bool),
        doc_ids=(
            None if bundle.get("doc_ids") is None else torch.as_tensor(bundle["doc_ids"], dtype=torch.long)
        ),
        k=k,
        similarity=similarity,
    )
