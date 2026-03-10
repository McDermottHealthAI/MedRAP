"""Retrieval backends for RAP pipeline composition.

This module provides retriever implementations for small in-memory corpora and
for dataset-backed corpora with attached nearest-neighbor indexes.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from datasets import Dataset
from torch import Tensor, nn

from .types import RetrieverOutput

_REFRESH_EXECUTOR = ThreadPoolExecutor(max_workers=1)


@dataclass(frozen=True)
class _DatasetSnapshot:
    """Dataset snapshot used for retrieval.

    Args:
        dataset: Hugging Face dataset with an attached nearest-neighbor index.
        index_name: Name of the attached index on ``dataset``.
    """

    dataset: Dataset
    index_name: str


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


class HFDatasetRetriever(Retriever):
    """Dataset-backed FAISS retriever.

    Uses a Hugging Face dataset as the document store and a FAISS index for
    nearest-neighbor retrieval. Retrieval always serves from the active
    snapshot.

    Args:
        dataset: Hugging Face dataset with an attached nearest-neighbor index.
        index_name: Name of the attached index on ``dataset``.
        doc_tokens_column: Column containing document token ids.
        doc_attention_mask_column: Column containing document attention masks.
        k: Number of documents to return per query.
        doc_ids_column: Optional column containing document ids.
        doc_key_embeddings_column: Optional column containing document key
            embeddings.
    """

    def __init__(
        self,
        *,
        dataset: Dataset,
        index_name: str,
        doc_tokens_column: str,
        doc_attention_mask_column: str,
        k: int = 1,
        doc_ids_column: str | None = None,
        doc_key_embeddings_column: str | None = None,
    ) -> None:
        super().__init__()

        if k < 1 or k > len(dataset):
            raise ValueError("k must be between 1 and the number of dataset rows")

        dataset_columns = set(dataset.column_names)
        required_columns = {doc_tokens_column, doc_attention_mask_column}
        missing_required = required_columns - dataset_columns
        if missing_required:
            raise ValueError(f"dataset is missing required columns: {sorted(missing_required)}")

        optional_columns = [doc_ids_column, doc_key_embeddings_column]
        missing_optional = [col for col in optional_columns if col is not None and col not in dataset_columns]
        if missing_optional:
            raise ValueError(f"dataset is missing optional columns: {sorted(missing_optional)}")

        self.k = k
        self._doc_tokens_column = doc_tokens_column
        self._doc_attention_mask_column = doc_attention_mask_column
        self._doc_ids_column = doc_ids_column
        self._doc_key_embeddings_column = doc_key_embeddings_column

        self._active_snapshot = self._build_snapshot(lambda: dataset, index_name=index_name)
        self._refresh_job: Future[_DatasetSnapshot] | None = None

    def _build_snapshot(
        self,
        build_dataset: Callable[[], Dataset],
        *,
        index_name: str,
    ) -> _DatasetSnapshot:
        dataset = build_dataset()
        try:
            dataset.get_index(index_name)
        except Exception as exc:
            raise ValueError(f"dataset does not have a FAISS index named {index_name!r}") from exc
        return _DatasetSnapshot(dataset=dataset, index_name=index_name)

    def start_refresh(self, *, build_dataset: Callable[[], Dataset]) -> bool:
        """Start a background refresh job.

        Args:
            build_dataset: Callable returning a fresh Hugging Face dataset with
                the attached index.

        Returns:
            ``False`` if a refresh job is already running.

        Examples:
            >>> retriever = HFDatasetRetriever(
            ...     dataset=FakeIndexedDataset(
            ...         doc_tokens=[[10, 11], [20, 21]],
            ...         doc_attention_mask=[[True, True], [True, False]],
            ...         doc_ids=[7, 8],
            ...     ),
            ...     index_name="embeddings",
            ...     doc_tokens_column="doc_tokens",
            ...     doc_attention_mask_column="doc_attention_mask",
            ...     doc_ids_column="doc_ids",
            ...     k=1,
            ... )
            >>> def build_dataset():
            ...     time.sleep(0.05)
            ...     return FakeIndexedDataset(
            ...         doc_tokens=[[30, 31], [40, 41]],
            ...         doc_attention_mask=[[True, True], [True, True]],
            ...         doc_ids=[17, 18],
            ...     )
            >>> retriever.start_refresh(build_dataset=build_dataset)
            True
            >>> retriever.start_refresh(build_dataset=build_dataset)
            False
            >>> time.sleep(0.1)
            >>> retriever.retrieve(torch.ones((1, 1, 4), dtype=torch.float32)).doc_ids.tolist()
            [[[17]]]
        """
        if self._refresh_job is not None:
            return False
        index_name = self._active_snapshot.index_name
        self._refresh_job = _REFRESH_EXECUTOR.submit(
            self._build_snapshot,
            build_dataset,
            index_name=index_name,
        )
        return True

    def _poll_refresh(self) -> bool:
        """Swap in a finished refresh snapshot if one is available."""
        if self._refresh_job is None:
            return False
        if not self._refresh_job.done():
            return False

        new_snapshot = self._refresh_job.result()
        try:
            new_snapshot.dataset.get_index(new_snapshot.index_name)
        except Exception as exc:
            self._refresh_job = None
            raise RuntimeError(
                f"refreshed dataset does not have a FAISS index named {new_snapshot.index_name!r}"
            ) from exc

        self._active_snapshot = new_snapshot
        self._refresh_job = None
        return True

    def _search_index(self, query_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """Search the active dataset index."""
        batch_size, n_retrieval_steps, d_ret = query_embeddings.shape
        snapshot = self._active_snapshot

        flat_queries = (
            query_embeddings.detach()
            .to(torch.float32)
            .cpu()
            .reshape(batch_size * n_retrieval_steps, d_ret)
            .numpy()
        )

        total_scores, total_indices = snapshot.dataset.search_batch(
            snapshot.index_name,
            flat_queries,
            k=self.k,
        )

        scores = torch.as_tensor(total_scores, dtype=torch.float32).reshape(
            batch_size,
            n_retrieval_steps,
            self.k,
        )
        row_indices = torch.as_tensor(total_indices, dtype=torch.long).reshape(
            batch_size,
            n_retrieval_steps,
            self.k,
        )
        return scores, row_indices

    def _materialize_output(
        self,
        *,
        row_indices: Tensor,
        scores: Tensor,
        output_device: torch.device,
    ) -> RetrieverOutput:
        """Materialize retrieved dataset rows into ``RetrieverOutput``."""
        if row_indices.ndim != 3:
            raise ValueError("row_indices must have shape (B, R, K)")
        if scores.shape != row_indices.shape:
            raise ValueError("scores must have the same shape as row_indices")
        if (row_indices < 0).any():
            raise RuntimeError("retrieval returned invalid dataset row indices")

        batch_size, n_retrieval_steps, k = row_indices.shape
        snapshot = self._active_snapshot
        flat_row_indices = row_indices.reshape(-1).tolist()
        rows = snapshot.dataset[flat_row_indices]

        doc_tokens = torch.as_tensor(
            rows[self._doc_tokens_column],
            dtype=torch.long,
            device=output_device,
        ).reshape(batch_size, n_retrieval_steps, k, -1)

        doc_attention_mask = torch.as_tensor(
            rows[self._doc_attention_mask_column],
            dtype=torch.bool,
            device=output_device,
        ).reshape(batch_size, n_retrieval_steps, k, -1)

        doc_ids = None
        if self._doc_ids_column is not None:
            doc_ids = torch.as_tensor(
                rows[self._doc_ids_column],
                dtype=torch.long,
                device=output_device,
            ).reshape(batch_size, n_retrieval_steps, k)

        doc_key_embeddings = None
        if self._doc_key_embeddings_column is not None:
            doc_key_embeddings = torch.as_tensor(
                rows[self._doc_key_embeddings_column],
                dtype=torch.float32,
                device=output_device,
            ).reshape(batch_size, n_retrieval_steps, k, -1)

        return RetrieverOutput(
            doc_tokens=doc_tokens,
            doc_attention_mask=doc_attention_mask,
            doc_scores=scores.to(output_device),
            doc_ids=doc_ids,
            doc_key_embeddings=doc_key_embeddings,
        )

    def retrieve(self, query_embeddings: Tensor) -> RetrieverOutput:
        """Retrieve top-k documents for queries of shape ``(B, R, D_ret)``."""
        if query_embeddings.ndim != 3:
            raise ValueError("query_embeddings must have shape (B, R, D_ret)")

        self._poll_refresh()

        scores, row_indices = self._search_index(query_embeddings)
        return self._materialize_output(
            row_indices=row_indices,
            scores=scores,
            output_device=query_embeddings.device,
        )


def load_in_memory_retriever(
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
        ...     retriever = load_in_memory_retriever(bundle_path=bundle_path, k=1)
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
