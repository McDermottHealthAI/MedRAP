"""Retrieval backends for RAP pipeline composition.

This module provides retriever implementations for small in-memory corpora and for dataset-backed corpora with
attached nearest-neighbor indexes.
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


@dataclass
class HFDatasetSnapshotBuilder:
    """Build fresh dataset snapshots for ``HFDatasetRetriever``.

    Builds a fresh dataset snapshot and does not modify the active snapshot in
    place.

    Args:
        source_dataset: Source Hugging Face dataset used to build snapshots.
        index_name: Name of the FAISS index to attach to built snapshots.
        key_embedding_column: Column containing rebuilt document key
            embeddings.
        encode_documents: Callable that returns one document key embedding per
            row in a batched dataset mapping input.
        batch_size: Batch size used when rebuilding the key embedding column.
    """

    source_dataset: Dataset
    index_name: str
    key_embedding_column: str
    encode_documents: Callable[[dict[str, list[object]], object | None], list[list[float]]]
    batch_size: int = 256

    def build_snapshot(self, refresh_context: object | None = None) -> _DatasetSnapshot:
        """Build a fresh indexed dataset snapshot.

        Args:
            refresh_context: Optional build input passed to
                ``encode_documents``.

        Returns:
            A fresh ``_DatasetSnapshot`` with rebuilt document key embeddings
            and a rebuilt FAISS index.

        Examples:
            >>> source = Dataset.from_dict(
            ...     {
            ...         "doc_tokens": [[10, 11], [20, 21]],
            ...         "doc_attention_mask": [[True, True], [True, False]],
            ...         "text": ["alpha", "beta"],
            ...     }
            ... )
            >>> original_add_faiss_index = Dataset.add_faiss_index
            >>> def fake_add_faiss_index(self, column, index_name):
            ...     def get_index(name):
            ...         if name != index_name:
            ...             raise KeyError(name)
            ...         return object()
            ...
            ...     self.get_index = get_index
            ...     return self
            >>> Dataset.add_faiss_index = fake_add_faiss_index
            >>> builder = HFDatasetSnapshotBuilder(
            ...     source_dataset=source,
            ...     index_name="retrieval",
            ...     key_embedding_column="doc_key_embeddings",
            ...     encode_documents=lambda batch, refresh_context: [[1.0, 0.0], [0.0, 1.0]],
            ... )
            >>> snapshot = builder.build_snapshot()
            >>> "doc_key_embeddings" in snapshot.dataset.column_names
            True
            >>> snapshot.index_name
            'retrieval'
            >>> Dataset.add_faiss_index = original_add_faiss_index
        """
        dataset = self.source_dataset
        if self.key_embedding_column in dataset.column_names:
            dataset = dataset.remove_columns(self.key_embedding_column)

        def add_key_embeddings(batch: dict[str, list[object]]) -> dict[str, list[list[float]]]:
            return {
                self.key_embedding_column: self.encode_documents(
                    batch,
                    refresh_context,
                )
            }

        dataset = dataset.map(
            add_key_embeddings,
            batched=True,
            batch_size=self.batch_size,
        )
        dataset.add_faiss_index(column=self.key_embedding_column, index_name=self.index_name)
        return _DatasetSnapshot(dataset=dataset, index_name=self.index_name)


class HFDatasetRetriever(Retriever):
    """Dataset-backed FAISS retriever.

    Uses a Hugging Face dataset as the document store and a FAISS index for
    nearest-neighbor retrieval. Retrieval always serves from the active
    snapshot.

    Args:
        active_snapshot: Active dataset snapshot used for retrieval.
        snapshot_builder: Optional builder used to construct refreshed
            snapshots.
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
        active_snapshot: _DatasetSnapshot,
        snapshot_builder: HFDatasetSnapshotBuilder | None = None,
        doc_tokens_column: str,
        doc_attention_mask_column: str,
        k: int = 1,
        doc_ids_column: str | None = None,
        doc_key_embeddings_column: str | None = None,
    ) -> None:
        super().__init__()

        self.k = k
        self._doc_tokens_column = doc_tokens_column
        self._doc_attention_mask_column = doc_attention_mask_column
        self._doc_ids_column = doc_ids_column
        self._doc_key_embeddings_column = doc_key_embeddings_column

        self._snapshot_builder = snapshot_builder
        self._active_snapshot = active_snapshot
        self._validate_snapshot(self._active_snapshot)
        if snapshot_builder is not None:
            self._refresh_executor = ThreadPoolExecutor(max_workers=1)
        else:
            self._refresh_executor = None
        self._refresh_job: Future[_DatasetSnapshot] | None = None

    def _validate_snapshot(self, snapshot: _DatasetSnapshot) -> None:
        dataset = snapshot.dataset

        if self.k < 1 or self.k > len(dataset):
            raise ValueError("k must be between 1 and the number of dataset rows")

        dataset_columns = set(dataset.column_names)
        required_columns = {self._doc_tokens_column, self._doc_attention_mask_column}
        missing_required = required_columns - dataset_columns
        if missing_required:
            raise ValueError(f"dataset is missing required columns: {sorted(missing_required)}")

        optional_columns = [self._doc_ids_column, self._doc_key_embeddings_column]
        missing_optional = [col for col in optional_columns if col is not None and col not in dataset_columns]
        if missing_optional:
            raise ValueError(f"dataset is missing optional columns: {sorted(missing_optional)}")

        try:
            dataset.get_index(snapshot.index_name)
        except Exception as exc:
            raise ValueError(f"dataset does not have a FAISS index named {snapshot.index_name!r}") from exc

    def start_refresh(self, *, refresh_context: object | None = None) -> bool:
        """Start a background refresh job.

        Args:
            refresh_context: Optional build input passed to the snapshot
                builder.

        Returns:
            ``False`` if a refresh job is already running.

        Examples:
            >>> initial_snapshot = _DatasetSnapshot(
            ...     dataset=FakeIndexedDataset(
            ...         doc_tokens=[[10, 11], [20, 21]],
            ...         doc_attention_mask=[[True, True], [True, False]],
            ...         doc_ids=[7, 8],
            ...     ),
            ...     index_name="embeddings",
            ... )
            >>> builder = FakeSnapshotBuilder(
            ...     snapshot=_DatasetSnapshot(
            ...         dataset=FakeIndexedDataset(
            ...             doc_tokens=[[30, 31], [40, 41]],
            ...             doc_attention_mask=[[True, True], [True, True]],
            ...             doc_ids=[17, 18],
            ...         ),
            ...         index_name="embeddings",
            ...     )
            ... )
            >>> retriever = HFDatasetRetriever(
            ...     active_snapshot=initial_snapshot,
            ...     snapshot_builder=builder,
            ...     doc_tokens_column="doc_tokens",
            ...     doc_attention_mask_column="doc_attention_mask",
            ...     doc_ids_column="doc_ids",
            ...     k=1,
            ... )
            >>> retriever.start_refresh()
            True
            >>> retriever.start_refresh()
            False
            >>> time.sleep(0.1)
            >>> retriever.retrieve(torch.ones((1, 1, 4), dtype=torch.float32)).doc_ids.tolist()
            [[[17]]]
        """
        if self._snapshot_builder is None or self._refresh_executor is None:
            raise RuntimeError("refresh is not available without a snapshot builder")
        if self._refresh_job is not None:
            return False
        self._refresh_job = self._refresh_executor.submit(
            self._snapshot_builder.build_snapshot,
            refresh_context,
        )
        return True

    def close(self) -> None:
        """Shut down the refresh executor."""
        if self._refresh_executor is not None:
            self._refresh_executor.shutdown(wait=False, cancel_futures=False)

    def _poll_refresh(self) -> bool:
        """Swap in a finished refresh snapshot if one is available."""
        if self._refresh_job is None:
            return False
        if not self._refresh_job.done():
            return False

        new_snapshot = self._refresh_job.result()
        self._validate_snapshot(new_snapshot)

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
        """Retrieve top-k documents for each query.

        Args:
            query_embeddings: Query tensor with shape ``(B, R, D_ret)``.

        Returns:
            ``RetrieverOutput`` with:
                - ``doc_tokens`` shaped ``(B, R, K, S_doc)``
                - ``doc_attention_mask`` shaped ``(B, R, K, S_doc)``
                - ``doc_scores`` shaped ``(B, R, K)``
                - optional ``doc_ids`` shaped ``(B, R, K)``
                - optional ``doc_key_embeddings`` shaped ``(B, R, K, D_ret)``

        Examples:
            >>> snapshot = _DatasetSnapshot(
            ...     dataset=FakeIndexedDataset(
            ...         doc_tokens=[[10, 11], [20, 21]],
            ...         doc_attention_mask=[[True, True], [True, False]],
            ...         doc_ids=[7, 8],
            ...     ),
            ...     index_name="embeddings",
            ... )
            >>> retriever = HFDatasetRetriever(
            ...     active_snapshot=snapshot,
            ...     doc_tokens_column="doc_tokens",
            ...     doc_attention_mask_column="doc_attention_mask",
            ...     doc_ids_column="doc_ids",
            ...     k=1,
            ... )
            >>> out = retriever.retrieve(torch.ones((1, 1, 4), dtype=torch.float32))
            >>> out.doc_ids.tolist()
            [[[7]]]
            >>> tuple(out.doc_tokens.shape)
            (1, 1, 1, 2)
        """
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
