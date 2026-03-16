"""Offline preparation helpers for Hugging Face retrieval datasets."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk


class OrderedFieldDocumentRenderer:
    """Render document text from ordered source fields.

    Args:
        fields: Source field names to render in order.
        separator: Separator inserted between rendered field fragments.
        include_field_names: Whether to prefix each field value with
            ``"{field}: "``.
    """

    def __init__(
        self,
        *,
        fields: Sequence[str],
        separator: str = "\n",
        include_field_names: bool = False,
    ) -> None:
        if not fields:
            raise ValueError("fields must contain at least one field name")
        self.fields = tuple(fields)
        self.separator = separator
        self.include_field_names = include_field_names

    def render(self, row: Mapping[str, object]) -> str:
        """Render a single dataset row into document text.

        Args:
            row: Mapping with at least the configured ``fields``.

        Returns:
            Rendered document text.

        Examples:
            >>> renderer = OrderedFieldDocumentRenderer(fields=["question", "answer"], separator=" | ")
            >>> renderer.render(
            ...     {
            ...         "question": "What is MedRAP?",
            ...         "answer": "A retrieval-augmented pretraining pipeline.",
            ...     }
            ... )
            'What is MedRAP? | A retrieval-augmented pretraining pipeline.'
            >>> named = OrderedFieldDocumentRenderer(fields=["question"], include_field_names=True)
            >>> named.render({"question": "What is MedRAP?"})
            'question: What is MedRAP?'
            >>> renderer.render({"question": "Missing answer"})  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: row is missing required fields: ['answer']
        """
        missing_fields = [field for field in self.fields if field not in row]
        if missing_fields:
            raise ValueError(f"row is missing required fields: {missing_fields}")

        fragments: list[str] = []
        for field in self.fields:
            value = "" if row[field] is None else str(row[field])
            fragments.append(f"{field}: {value}" if self.include_field_names else value)
        return self.separator.join(fragments)


def load_hf_dataset_source(
    *,
    path: str,
    split: str,
    name: str | None = None,
    data_files: str | list[str] | None = None,
) -> Dataset:
    """Load a Hugging Face dataset split via :func:`datasets.load_dataset`.

    Args:
        path: Dataset path, builder name, or local data file path supported by
            :func:`datasets.load_dataset`.
        split: Dataset split to load.
        name: Optional dataset config name.
        data_files: Optional local data files passed to
            :func:`datasets.load_dataset`.

    Returns:
        Loaded dataset split.
    """
    dataset = load_dataset(path=path, name=name, split=split, data_files=data_files)
    if not isinstance(dataset, Dataset):
        raise TypeError("load_dataset must return a datasets.Dataset for retrieval preparation")
    return dataset


def load_hf_dataset_from_disk(*, dataset_path: str | Path) -> Dataset:
    """Load a saved Hugging Face dataset artifact from disk.

    Args:
        dataset_path: Directory previously written by
            :meth:`datasets.Dataset.save_to_disk`.

    Returns:
        Loaded dataset.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     dataset = Dataset.from_dict({"text": ["alpha", "beta"]})
        ...     dataset.save_to_disk(tmpdir)
        ...     loaded = load_hf_dataset_from_disk(dataset_path=tmpdir)
        ...     list(loaded["text"])
        ['alpha', 'beta']
    """
    dataset = load_from_disk(str(dataset_path))
    if not isinstance(dataset, Dataset):
        raise TypeError("load_from_disk must return a datasets.Dataset for retrieval preparation")
    return dataset


def load_hf_tokenizer(*, model_name: str) -> Any:
    """Load a Hugging Face tokenizer.

    Args:
        model_name: Model or tokenizer identifier.

    Returns:
        Tokenizer instance exposing ``__call__`` with standard Hugging Face
        tokenization arguments.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def load_sentence_transformer(*, model_name: str, device: str = "cpu") -> Any:
    """Load a sentence-transformers embedding model.

    Args:
        model_name: Sentence-transformers model identifier.
        device: Device string passed to ``SentenceTransformer``.

    Returns:
        Sentence transformer model exposing ``encode``.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def prepare_retrieval_dataset(
    *,
    dataset: Dataset,
    renderer: OrderedFieldDocumentRenderer,
    tokenizer: Any,
    embedder: Any,
    output_dir: str | Path,
    doc_text_column: str = "doc_text",
    doc_tokens_column: str = "doc_tokens",
    doc_attention_mask_column: str = "doc_attention_mask",
    doc_key_embeddings_column: str = "doc_key_embeddings",
    doc_ids_column: str = "doc_ids",
    source_id_column: str | None = None,
    index_name: str = "retrieval",
    max_length: int = 512,
    tokenization_batch_size: int = 256,
    embedding_batch_size: int = 256,
    string_factory: str | None = None,
) -> Path:
    """Prepare and save a static retrieval dataset artifact.

    Args:
        dataset: Source Hugging Face dataset.
        renderer: Renderer producing document text from source rows.
        tokenizer: Tokenizer used to produce ``doc_tokens`` and
            ``doc_attention_mask``.
        embedder: Embedder used to produce ``doc_key_embeddings`` from
            ``doc_text``.
        output_dir: Output directory for the saved dataset artifact.
        doc_text_column: Output text column name.
        doc_tokens_column: Output token-id column name.
        doc_attention_mask_column: Output attention-mask column name.
        doc_key_embeddings_column: Output key-embedding column name.
        doc_ids_column: Output document-id column name.
        source_id_column: Optional source column copied into ``doc_ids``. When
            provided, values must be integer-like.
        index_name: Saved FAISS index name.
        max_length: Tokenizer max sequence length.
        tokenization_batch_size: Batch size used for tokenization mapping.
        embedding_batch_size: Batch size passed to the embedder.
        string_factory: Optional FAISS ``string_factory`` passed to
            :meth:`datasets.Dataset.add_faiss_index`.

    Returns:
        Output directory containing the saved dataset artifact and FAISS index
        sidecar.
    """

    def _coerce_doc_id(value: object, *, row_index: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{doc_ids_column} values must be integer-like; got {value!r} at row {row_index}"
            ) from exc

    def _render_row(row: Mapping[str, object], idx: int) -> dict[str, object]:
        rendered: dict[str, object] = {doc_text_column: renderer.render(row)}
        rendered[doc_ids_column] = (
            _coerce_doc_id(row[source_id_column], row_index=idx) if source_id_column is not None else idx
        )
        return rendered

    prepared = dataset.map(
        _render_row,
        with_indices=True,
        desc="Rendering retrieval documents",
    )

    def _tokenize_batch(batch: Mapping[str, object]) -> dict[str, object]:
        texts = cast("list[str]", batch[doc_text_column])
        encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        return {
            doc_tokens_column: encoded["input_ids"],
            doc_attention_mask_column: encoded["attention_mask"],
        }

    prepared = prepared.map(
        _tokenize_batch,
        batched=True,
        batch_size=tokenization_batch_size,
        desc="Tokenizing retrieval documents",
    )

    def _embed_batch(batch: Mapping[str, object]) -> dict[str, object]:
        texts = cast("list[str]", batch[doc_text_column])
        embeddings = embedder.encode(
            texts,
            batch_size=embedding_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return {doc_key_embeddings_column: np.asarray(embeddings, dtype=np.float32).tolist()}

    prepared = prepared.map(
        _embed_batch,
        batched=True,
        batch_size=embedding_batch_size,
        desc="Embedding retrieval documents",
    )

    prepared.add_faiss_index(
        column=doc_key_embeddings_column,
        index_name=index_name,
        string_factory=string_factory,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prepared.save_faiss_index(index_name, output_path / f"{index_name}.faiss")
    prepared.drop_index(index_name)
    prepared.save_to_disk(str(output_path))
    return output_path
