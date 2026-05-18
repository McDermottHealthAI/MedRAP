"""Retrieval backends for RAP pipeline composition.

This module provides retriever implementations for small in-memory corpora and for prepared Hugging Face
datasets with attached nearest-neighbor indexes.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from torch import Tensor, nn

from .types import RetrieverOutput


def _move_tensors_to_device(device: torch.device, *tensors: Tensor) -> tuple[Tensor, ...]:
    """Return tensors on ``device`` while preserving the no-op fast path.

    Examples:
        >>> cpu_tensor = torch.tensor([1, 2])
        >>> (same_device,) = _move_tensors_to_device(torch.device("cpu"), cpu_tensor)
        >>> same_device is cpu_tensor
        True
        >>> (moved,) = _move_tensors_to_device(torch.device("meta"), cpu_tensor)
        >>> moved.device.type
        'meta'
    """
    return tuple(tensor if tensor.device == device else tensor.to(device) for tensor in tensors)


def _apply_retrieval_ablation(
    *,
    ablation_mode: str,
    dataset_num_rows: int,
    scores: Tensor,
    row_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply configured retrieval ablation to searched row ids.

    Args:
        ablation_mode: Retrieval ablation mode, one of ``"none"`` or
            ``"random_docs"``.
        dataset_num_rows: Number of rows in the retrieval dataset.
        scores: Retrieval scores with shape ``(B, R, K)``.
        row_indices: Retrieved dataset row indices with shape ``(B, R, K)``.

    Returns:
        Possibly ablated ``(scores, row_indices)``. ``random_docs`` returns
        uniform random row ids sampled with replacement and zero scores.

    Examples:
        >>> scores = torch.FloatTensor([[[1.0]], [[2.0]]])
        >>> rows = torch.LongTensor([[[10]], [[20]]])
        >>> out_scores, out_rows = _apply_retrieval_ablation(
        ...     ablation_mode="none",
        ...     dataset_num_rows=100,
        ...     scores=scores,
        ...     row_indices=rows,
        ... )
        >>> torch.equal(out_scores, scores), torch.equal(out_rows, rows)
        (True, True)

        >>> _ = torch.manual_seed(1)
        >>> out_scores, out_rows = _apply_retrieval_ablation(
        ...     ablation_mode="random_docs",
        ...     dataset_num_rows=100,
        ...     scores=scores,
        ...     row_indices=rows,
        ... )
        >>> out_rows.shape == rows.shape and out_rows.max() < 100 and out_rows.min() >= 0
        tensor(True)
        >>> torch.equal(out_scores, torch.zeros_like(scores))
        True

        >>> _apply_retrieval_ablation(
        ...     ablation_mode="bad",
        ...     dataset_num_rows=100,
        ...     scores=scores,
        ...     row_indices=rows,
        ... )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        RuntimeError: Unsupported retrieval ablation mode: 'bad'
    """
    if ablation_mode == "none":
        return scores, row_indices
    if ablation_mode == "random_docs":
        random_rows = torch.randint(
            dataset_num_rows,
            row_indices.shape,
            device=row_indices.device,
            dtype=row_indices.dtype,
        )
        return torch.zeros_like(scores), random_rows
    raise RuntimeError(f"Unsupported retrieval ablation mode: {ablation_mode!r}")


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
            >>> meta_out = retriever.retrieve(torch.ones((1, 1, 2), device="meta"))
            >>> meta_out.doc_tokens.device.type
            'meta'
            >>> meta_out.doc_attention_mask.device.type
            'meta'
            >>> meta_out.doc_scores.device.type
            'meta'
            >>> meta_out.doc_ids.device.type
            'meta'
            >>> meta_out.doc_key_embeddings.device.type
            'meta'
        """
        if query_embeddings.ndim != 3:
            raise ValueError("query_embeddings must have shape (B, R, D_ret)")
        doc_key_embeddings, doc_tokens, doc_attention_mask, doc_ids = _move_tensors_to_device(
            query_embeddings.device,
            self._doc_key_embeddings,
            self._doc_tokens,
            self._doc_attention_mask,
            self._doc_ids,
        )

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

    Uses a prepared Hugging Face dataset as the document store and an attached
    FAISS index for nearest-neighbor retrieval.

    Args:
        dataset: Prepared dataset containing retrieval payload columns and an
            attached FAISS index.
        index_name: Name of the attached FAISS index on ``dataset``.
        doc_tokens_column: Column containing document token ids.
        doc_attention_mask_column: Column containing document attention masks.
        k: Number of documents to return per query.
        doc_ids_column: Optional column containing document ids.
        doc_key_embeddings_column: Optional column containing document key
            embeddings.
        ablation_mode: Retrieval ablation mode. ``"none"`` returns normal
            nearest-neighbor results. ``"random_docs"`` replaces
            nearest-neighbor row ids with uniform random corpus row ids sampled
            with replacement, breaking patient-document alignment while keeping
            the same payload materialization path.
        cache_payloads: Whether to cache retrieval payload columns as tensors at
            construction time. This avoids Hugging Face Dataset row
            materialization in the per-batch retrieval hot path.
        payload_cache_device: Device for cached payload tensors. ``None`` means
            CPU. Use ``"cuda:0"`` only when the payload corpus fits in GPU
            memory.

    Cached payloads preserve the same output shapes as uncached retrieval:
    ``doc_tokens`` and ``doc_attention_mask`` are ``(B, R, K, S_doc)``,
    ``doc_ids`` is ``(B, R, K)``, and ``doc_key_embeddings`` is
    ``(B, R, K, D_ret)`` when configured.
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
        ablation_mode: str = "none",
        cache_payloads: bool = False,
        payload_cache_device: str | int | torch.device | None = None,
    ) -> None:
        super().__init__()

        if ablation_mode not in {"none", "random_docs"}:
            raise ValueError("ablation_mode must be one of: 'none', 'random_docs'")
        self.k = k
        self.ablation_mode = ablation_mode
        self._doc_tokens_column = doc_tokens_column
        self._doc_attention_mask_column = doc_attention_mask_column
        self._doc_ids_column = doc_ids_column
        self._doc_key_embeddings_column = doc_key_embeddings_column
        self._dataset = dataset
        self._dataset_num_rows = len(dataset)
        self._index_name = index_name
        self._cached_doc_tokens: Tensor | None = None
        self._cached_doc_attention_mask: Tensor | None = None
        self._cached_doc_ids: Tensor | None = None
        self._cached_doc_key_embeddings: Tensor | None = None
        self._validate_dataset()
        if cache_payloads:
            self._cache_payloads(payload_cache_device)

    @staticmethod
    def _resolve_cache_device(device: str | int | torch.device | None) -> torch.device:
        """Resolve payload cache device names.

        Examples:
            >>> HFDatasetRetriever._resolve_cache_device(None)
            device(type='cpu')
            >>> HFDatasetRetriever._resolve_cache_device("cpu")
            device(type='cpu')
            >>> HFDatasetRetriever._resolve_cache_device(torch.device("cpu"))
            device(type='cpu')
            >>> HFDatasetRetriever._resolve_cache_device(2)
            device(type='cuda', index=2)
        """
        if device is None:
            return torch.device("cpu")
        if isinstance(device, torch.device):
            return device
        if isinstance(device, int):
            return torch.device("cuda", device)
        return torch.device(device)

    def _cache_dataset_column(
        self,
        column: str,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Convert one fixed-shape dataset column to a tensor cache.

        Args:
            column: Dataset column to cache.
            dtype: Tensor dtype for the cached column.
            device: Device that stores the cache.

        Returns:
            Tensor containing all rows from ``column`` on ``device``. The first
            dimension is always ``N_docs``.

        Examples:
            >>> class IterableColumn:
            ...     def __iter__(self):
            ...         return iter([[1, 2], [3, 4]])
            >>> retriever = object.__new__(HFDatasetRetriever)
            >>> retriever._dataset = {"doc_tokens": IterableColumn()}
            >>> cached = retriever._cache_dataset_column(
            ...     "doc_tokens",
            ...     dtype=torch.long,
            ...     device=torch.device("cpu"),
            ... )
            >>> cached.tolist()
            [[1, 2], [3, 4]]
        """
        values = self._dataset[column]
        try:
            tensor = torch.as_tensor(values, dtype=dtype)
        except (TypeError, ValueError):
            tensor = torch.as_tensor(list(values), dtype=dtype)
        return tensor.to(device)

    def _cache_payloads(self, device: str | int | torch.device | None = None) -> None:
        """Cache configured retrieval payload columns as indexable tensors.

        Args:
            device: Target cache device. ``None`` stores payloads on CPU.
        """
        cache_device = self._resolve_cache_device(device)
        self._cached_doc_tokens = self._cache_dataset_column(
            self._doc_tokens_column,
            dtype=torch.long,
            device=cache_device,
        )
        self._cached_doc_attention_mask = self._cache_dataset_column(
            self._doc_attention_mask_column,
            dtype=torch.bool,
            device=cache_device,
        )
        if self._doc_ids_column is not None:
            self._cached_doc_ids = self._cache_dataset_column(
                self._doc_ids_column,
                dtype=torch.long,
                device=cache_device,
            )
        if self._doc_key_embeddings_column is not None:
            self._cached_doc_key_embeddings = self._cache_dataset_column(
                self._doc_key_embeddings_column,
                dtype=torch.float32,
                device=cache_device,
            )

    def _validate_dataset(self) -> None:
        """Validate dataset columns and attached index.

        Raises:
            ValueError: If ``k`` is out of range, required payload columns are
                missing, configured optional columns are missing, or the FAISS
                index is not attached.

        Examples:
            >>> from datasets import Dataset
            >>> dataset = Dataset.from_dict(
            ...     {
            ...         "doc_tokens": [[10, 11], [20, 21]],
            ...         "doc_attention_mask": [[1, 1], [1, 0]],
            ...         "index_embeddings": [[1.0, 0.0], [0.0, 1.0]],
            ...     }
            ... )
            >>> dataset.add_faiss_index(column="index_embeddings", index_name="retrieval")
            Dataset({
                ...
            })
            >>> HFDatasetRetriever(
            ...     dataset=dataset,
            ...     index_name="retrieval",
            ...     doc_tokens_column="doc_tokens",
            ...     doc_attention_mask_column="doc_attention_mask",
            ...     k=3,
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: k must be between 1 and the number of dataset rows

            >>> missing_required = Dataset.from_dict(
            ...     {
            ...         "doc_attention_mask": [[1, 1]],
            ...         "index_embeddings": [[1.0, 0.0]],
            ...     }
            ... )
            >>> missing_required.add_faiss_index(column="index_embeddings", index_name="retrieval")
            Dataset({
                ...
            })
            >>> HFDatasetRetriever(
            ...     dataset=missing_required,
            ...     index_name="retrieval",
            ...     doc_tokens_column="doc_tokens",
            ...     doc_attention_mask_column="doc_attention_mask",
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: dataset is missing required columns: ['doc_tokens']

            >>> missing_optional = Dataset.from_dict(
            ...     {
            ...         "doc_tokens": [[10, 11]],
            ...         "doc_attention_mask": [[1, 1]],
            ...         "index_embeddings": [[1.0, 0.0]],
            ...     }
            ... )
            >>> missing_optional.add_faiss_index(column="index_embeddings", index_name="retrieval")
            Dataset({
                ...
            })
            >>> HFDatasetRetriever(
            ...     dataset=missing_optional,
            ...     index_name="retrieval",
            ...     doc_tokens_column="doc_tokens",
            ...     doc_attention_mask_column="doc_attention_mask",
            ...     doc_ids_column="doc_ids",
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: dataset is missing optional columns: ['doc_ids']

            >>> no_index = Dataset.from_dict(
            ...     {
            ...         "doc_tokens": [[10, 11]],
            ...         "doc_attention_mask": [[1, 1]],
            ...     }
            ... )
            >>> HFDatasetRetriever(
            ...     dataset=no_index,
            ...     index_name="retrieval",
            ...     doc_tokens_column="doc_tokens",
            ...     doc_attention_mask_column="doc_attention_mask",
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: dataset does not have a FAISS index named 'retrieval'
        """
        dataset = self._dataset

        if self.k < 1 or self.k > len(dataset):
            raise ValueError("k must be between 1 and the number of dataset rows")

        dataset_columns = set(dataset.column_names)
        required_columns = {self._doc_tokens_column, self._doc_attention_mask_column}
        missing_required = required_columns - dataset_columns
        if missing_required:
            raise ValueError(f"dataset is missing required columns: {sorted(missing_required)}")

        optional_columns = [self._doc_ids_column, self._doc_key_embeddings_column]
        missing_optional = [col for col in optional_columns if col is not None if col not in dataset_columns]
        if missing_optional:
            raise ValueError(f"dataset is missing optional columns: {sorted(missing_optional)}")

        try:
            dataset.get_index(self._index_name)
        except Exception as exc:
            raise ValueError(f"dataset does not have a FAISS index named {self._index_name!r}") from exc

    def _search_index(self, query_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """Search the attached dataset FAISS index.

        Args:
            query_embeddings: Query tensor with shape ``(B, R, D_ret)``.

        Returns:
            Tuple ``(scores, row_indices)`` where both tensors have shape
            ``(B, R, K)``.
        """
        batch_size, n_retrieval_steps, d_ret = query_embeddings.shape

        flat_queries = (
            query_embeddings.detach()
            .to(torch.float32)
            .cpu()
            .reshape(batch_size * n_retrieval_steps, d_ret)
            .numpy()
        )

        total_scores, total_indices = self._dataset.search_batch(
            self._index_name,
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
        """Materialize retrieved dataset rows into ``RetrieverOutput``.

        Args:
            row_indices: Retrieved dataset row indices with shape ``(B, R, K)``.
            scores: Retrieval scores with shape ``(B, R, K)``.
            output_device: Device to place the materialized tensors on.

        Returns:
            ``RetrieverOutput`` with:
                - ``doc_tokens`` shaped ``(B, R, K, S_doc)``
                - ``doc_attention_mask`` shaped ``(B, R, K, S_doc)``
                - ``doc_scores`` shaped ``(B, R, K)``
                - ``doc_ids`` shaped ``(B, R, K)`` (from ``doc_ids_column`` if
                  configured, otherwise the FAISS dataset row indices)
                - optional ``doc_key_embeddings`` shaped ``(B, R, K, D_ret)``
        """
        if self._cached_doc_tokens is not None:
            return self._materialize_cached_output(
                row_indices=row_indices,
                scores=scores,
                output_device=output_device,
            )

        batch_size, n_retrieval_steps, k = row_indices.shape
        flat_row_indices = row_indices.reshape(-1).tolist()
        rows = self._dataset[flat_row_indices]

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

        if self._doc_ids_column is not None:
            doc_ids = torch.as_tensor(
                rows[self._doc_ids_column],
                dtype=torch.long,
                device=output_device,
            ).reshape(batch_size, n_retrieval_steps, k)
        else:
            # Fall back to FAISS row indices (the canonical dataset row id).
            doc_ids = row_indices.to(output_device)

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

    def _take_cached_rows(self, cache: Tensor, row_indices: Tensor, output_device: torch.device) -> Tensor:
        """Gather cached payload rows and return them on ``output_device``.

        Args:
            cache: Cached tensor with first dimension ``N_docs``.
            row_indices: Retrieved row ids with shape ``(B, R, K)``.
            output_device: Device for the returned tensor.

        Returns:
            Gathered rows with flattened leading dimension ``B * R * K`` and
            remaining dimensions inherited from ``cache``.
        """
        flat_row_indices = row_indices.reshape(-1).to(cache.device)
        values = cache.index_select(0, flat_row_indices)
        return values.to(output_device)

    def _materialize_cached_output(
        self,
        *,
        row_indices: Tensor,
        scores: Tensor,
        output_device: torch.device,
    ) -> RetrieverOutput:
        """Materialize retrieved rows from cached tensor payloads.

        Args:
            row_indices: Retrieved dataset row indices with shape ``(B, R, K)``.
            scores: Retrieval scores with shape ``(B, R, K)``.
            output_device: Device to place the materialized tensors on.

        Returns:
            ``RetrieverOutput`` with:
                - ``doc_tokens`` shaped ``(B, R, K, S_doc)``
                - ``doc_attention_mask`` shaped ``(B, R, K, S_doc)``
                - ``doc_scores`` shaped ``(B, R, K)``
                - ``doc_ids`` shaped ``(B, R, K)``
                - optional ``doc_key_embeddings`` shaped ``(B, R, K, D_ret)``

        Examples:
            >>> retriever = object.__new__(HFDatasetRetriever)
            >>> retriever._cached_doc_tokens = None
            >>> retriever._cached_doc_attention_mask = None
            >>> retriever._materialize_cached_output(
            ...     row_indices=torch.LongTensor([[[0]]]),
            ...     scores=torch.FloatTensor([[[1.0]]]),
            ...     output_device=torch.device("cpu"),
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            RuntimeError: Cached payloads are not initialized.
        """
        if self._cached_doc_tokens is None or self._cached_doc_attention_mask is None:
            raise RuntimeError("Cached payloads are not initialized.")

        batch_size, n_retrieval_steps, k = row_indices.shape

        doc_tokens = self._take_cached_rows(
            self._cached_doc_tokens,
            row_indices,
            output_device,
        ).reshape(batch_size, n_retrieval_steps, k, -1)
        doc_attention_mask = self._take_cached_rows(
            self._cached_doc_attention_mask,
            row_indices,
            output_device,
        ).reshape(batch_size, n_retrieval_steps, k, -1)

        if self._cached_doc_ids is not None:
            doc_ids = self._take_cached_rows(
                self._cached_doc_ids,
                row_indices,
                output_device,
            ).reshape(batch_size, n_retrieval_steps, k)
        else:
            doc_ids = row_indices.to(output_device)

        doc_key_embeddings = None
        if self._cached_doc_key_embeddings is not None:
            doc_key_embeddings = self._take_cached_rows(
                self._cached_doc_key_embeddings,
                row_indices,
                output_device,
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
            query_embeddings: Query tensor with shape ``(B, R, D_ret)``. Queries
            maybe on any PyTorch device.

        Returns:
            ``RetrieverOutput`` on the same device as
            ``query_embeddings`` with:
                - ``doc_tokens`` shaped ``(B, R, K, S_doc)``
                - ``doc_attention_mask`` shaped ``(B, R, K, S_doc)``
                - ``doc_scores`` shaped ``(B, R, K)``
                - optional ``doc_ids`` shaped ``(B, R, K)``
                - optional ``doc_key_embeddings`` shaped ``(B, R, K, D_ret)``

        Search is performed through the Hugging Face dataset index. Query
        embeddings are moved to CPU/NumPy internally for search, and retrieved
        tensors are materialized back on the query device.

        Examples:
            >>> from datasets import Dataset
            >>> dataset = Dataset.from_dict(
            ...     {
            ...         "doc_tokens": [[10, 11], [20, 21]],
            ...         "doc_attention_mask": [[1, 1], [1, 0]],
            ...         "doc_ids": [7, 8],
            ...         "index_embeddings": [[1.0, 0.0], [0.0, 1.0]],
            ...         "doc_key_embeddings": [[1.0, 0.0], [0.0, 1.0]],
            ...     }
            ... )
            >>> dataset.add_faiss_index(column="index_embeddings", index_name="retrieval")
            Dataset({
                ...
            })
            >>> retriever = HFDatasetRetriever(
            ...     dataset=dataset,
            ...     index_name="retrieval",
            ...     doc_tokens_column="doc_tokens",
            ...     doc_attention_mask_column="doc_attention_mask",
            ...     doc_ids_column="doc_ids",
            ...     doc_key_embeddings_column="doc_key_embeddings",
            ...     k=1,
            ... )
            >>> out = retriever(torch.FloatTensor([[[1.0, 0.0]], [[0.0, 1.0]]]))
            >>> tuple(out.doc_tokens.shape)
            (2, 1, 1, 2)
            >>> out.doc_ids.tolist()
            [[[7]], [[8]]]
            >>> tuple(out.doc_key_embeddings.shape)
            (2, 1, 1, 2)
            >>> retriever.retrieve(torch.ones(2, 2))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: query_embeddings must have shape (B, R, D_ret)

            >>> retriever_no_ids = HFDatasetRetriever(
            ...     dataset=dataset,
            ...     index_name="retrieval",
            ...     doc_tokens_column="doc_tokens",
            ...     doc_attention_mask_column="doc_attention_mask",
            ...     k=1,
            ... )
            >>> out_no_ids = retriever_no_ids(torch.FloatTensor([[[1.0, 0.0]], [[0.0, 1.0]]]))
            >>> out_no_ids.doc_ids.tolist()
            [[[0]], [[1]]]

            >>> cached = HFDatasetRetriever(
            ...     dataset=dataset,
            ...     index_name="retrieval",
            ...     doc_tokens_column="doc_tokens",
            ...     doc_attention_mask_column="doc_attention_mask",
            ...     doc_ids_column="doc_ids",
            ...     doc_key_embeddings_column="doc_key_embeddings",
            ...     k=1,
            ...     cache_payloads=True,
            ... )
            >>> cached_out = cached(torch.FloatTensor([[[1.0, 0.0]], [[0.0, 1.0]]]))
            >>> torch.equal(cached_out.doc_tokens, out.doc_tokens)
            True
            >>> torch.equal(cached_out.doc_attention_mask, out.doc_attention_mask)
            True
            >>> torch.equal(cached_out.doc_ids, out.doc_ids)
            True
            >>> torch.equal(cached_out.doc_key_embeddings, out.doc_key_embeddings)
            True
            >>> cached._cached_doc_tokens.device.type
            'cpu'

            >>> cached_no_ids = HFDatasetRetriever(
            ...     dataset=dataset,
            ...     index_name="retrieval",
            ...     doc_tokens_column="doc_tokens",
            ...     doc_attention_mask_column="doc_attention_mask",
            ...     k=1,
            ...     cache_payloads=True,
            ... )
            >>> out_cached_no_ids = cached_no_ids(torch.FloatTensor([[[0.0, 1.0]]]))
            >>> out_cached_no_ids.doc_ids.tolist()
            [[[1]]]
            >>> out_cached_no_ids.doc_key_embeddings is None
            True

            >>> HFDatasetRetriever(
            ...     dataset=dataset,
            ...     index_name="retrieval",
            ...     doc_tokens_column="doc_tokens",
            ...     doc_attention_mask_column="doc_attention_mask",
            ...     ablation_mode="bad",
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: ablation_mode must be one of: 'none', 'random_docs'
        """
        if query_embeddings.ndim != 3:
            raise ValueError("query_embeddings must have shape (B, R, D_ret)")

        scores, row_indices = self._search_index(query_embeddings)
        scores, row_indices = _apply_retrieval_ablation(
            ablation_mode=self.ablation_mode,
            dataset_num_rows=self._dataset_num_rows,
            scores=scores,
            row_indices=row_indices,
        )
        return self._materialize_output(
            row_indices=row_indices,
            scores=scores,
            output_device=query_embeddings.device,
        )


def load_hf_dataset_retriever(
    *,
    dataset_path: str,
    index_name: str,
    doc_tokens_column: str,
    doc_attention_mask_column: str,
    k: int = 1,
    doc_ids_column: str | None = None,
    doc_key_embeddings_column: str | None = None,
    index_path: str | None = None,
    device: int | list[int] | None = None,
    ablation_mode: str = "none",
    cache_payloads: bool = False,
    payload_cache_device: str | int | torch.device | None = None,
) -> HFDatasetRetriever:
    """Load a static dataset-backed retriever from a saved artifact.

    Args:
        dataset_path: Directory written by
            :meth:`datasets.Dataset.save_to_disk`.
        index_name: Name of the saved FAISS index.
        doc_tokens_column: Column containing document token ids.
        doc_attention_mask_column: Column containing document attention masks.
        k: Number of documents to return per query.
        doc_ids_column: Optional column containing document ids.
        doc_key_embeddings_column: Optional column containing document key
            embeddings.
        index_path: Optional explicit path to the saved FAISS index file. When
            omitted, defaults to ``{dataset_path}/{index_name}.faiss``.
        device: Optional FAISS device passed to
            :meth:`datasets.Dataset.load_faiss_index`. ``None`` keeps the index
            on CPU; ``0`` uses GPU 0; ``-1`` uses all GPUs.
        ablation_mode: Retrieval ablation mode passed to
            :class:`HFDatasetRetriever`.
        cache_payloads: Whether to cache retrieval payload columns as tensors at
            load time.
        payload_cache_device: Device for cached payload tensors. ``None`` means
            CPU.

    Returns:
        Configured :class:`HFDatasetRetriever`.

    Examples:
        >>> from datasets import Dataset
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     dataset = Dataset.from_dict(
        ...         {
        ...             "doc_tokens": [[10, 11], [20, 21]],
        ...             "doc_attention_mask": [[1, 1], [1, 0]],
        ...             "doc_ids": [7, 8],
        ...             "doc_key_embeddings": [[1.0, 0.0], [0.0, 1.0]],
        ...         }
        ...     )
        ...     _ = dataset.add_faiss_index(column="doc_key_embeddings", index_name="retrieval")
        ...     dataset.save_faiss_index("retrieval", f"{tmpdir}/retrieval.faiss")
        ...     _ = dataset.drop_index("retrieval")
        ...     _ = dataset.save_to_disk(tmpdir)
        ...     retriever = load_hf_dataset_retriever(
        ...         dataset_path=tmpdir,
        ...         index_name="retrieval",
        ...         doc_tokens_column="doc_tokens",
        ...         doc_attention_mask_column="doc_attention_mask",
        ...         doc_ids_column="doc_ids",
        ...         doc_key_embeddings_column="doc_key_embeddings",
        ...         k=1,
        ...         index_path=f"{tmpdir}/retrieval.faiss",
        ...     )
        ...     tuple(retriever.retrieve(torch.FloatTensor([[[1.0, 0.0]]])).doc_tokens.shape)
        (1, 1, 1, 2)
    """
    dataset = load_from_disk(dataset_path)
    resolved_index_path = (
        index_path if index_path is not None else str(Path(dataset_path) / f"{index_name}.faiss")
    )
    try:
        dataset.load_faiss_index(index_name, resolved_index_path, device=device)
    except Exception as exc:
        if device is not None:
            raise RuntimeError(
                "FAISS GPU retrieval was requested with device="
                f"{device!r}, but the installed FAISS/datasets stack could not load the index on GPU. "
                "Install a CUDA-enabled FAISS build compatible with this environment, or set "
                "retriever.device=null to use CPU retrieval."
            ) from exc
        raise
    return HFDatasetRetriever(
        dataset=dataset,
        index_name=index_name,
        doc_tokens_column=doc_tokens_column,
        doc_attention_mask_column=doc_attention_mask_column,
        k=k,
        doc_ids_column=doc_ids_column,
        doc_key_embeddings_column=doc_key_embeddings_column,
        ablation_mode=ablation_mode,
        cache_payloads=cache_payloads,
        payload_cache_device=payload_cache_device,
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
