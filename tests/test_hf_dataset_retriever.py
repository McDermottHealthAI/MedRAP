import pytest
import torch
from datasets import Dataset

from medrap.retrievers import HFDatasetRetriever


def _build_indexed_dataset(
    *,
    include_doc_ids: bool = True,
    include_doc_key_embeddings: bool = False,
    index_name: str = "retrieval",
) -> Dataset:
    columns: dict[str, object] = {
        "doc_tokens": [[10, 11], [20, 21], [30, 31]],
        "doc_attention_mask": [[1, 1], [1, 1], [1, 0]],
        "index_embeddings": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    }
    if include_doc_ids:
        columns["doc_ids"] = [7, 8, 9]
    if include_doc_key_embeddings:
        columns["doc_key_embeddings"] = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    dataset = Dataset.from_dict(columns)
    dataset.add_faiss_index(column="index_embeddings", index_name=index_name)
    return dataset


def test_retrieve_returns_retriever_output() -> None:
    dataset = _build_indexed_dataset(include_doc_key_embeddings=True)
    retriever = HFDatasetRetriever(
        dataset=dataset,
        index_name="retrieval",
        doc_tokens_column="doc_tokens",
        doc_attention_mask_column="doc_attention_mask",
        doc_ids_column="doc_ids",
        doc_key_embeddings_column="doc_key_embeddings",
        k=2,
    )

    out = retriever.retrieve(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32))

    assert tuple(out.doc_tokens.shape) == (2, 1, 2, 2)
    assert tuple(out.doc_attention_mask.shape) == (2, 1, 2, 2)
    assert tuple(out.doc_scores.shape) == (2, 1, 2)
    assert out.doc_ids is not None
    assert tuple(out.doc_ids.shape) == (2, 1, 2)
    assert out.doc_ids[:, :, 0].tolist() == [[7], [8]]
    assert out.doc_key_embeddings is not None
    assert tuple(out.doc_key_embeddings.shape) == (2, 1, 2, 2)


def test_materialize_output_validates_shapes_and_indices() -> None:
    retriever = HFDatasetRetriever(
        dataset=_build_indexed_dataset(),
        index_name="retrieval",
        doc_tokens_column="doc_tokens",
        doc_attention_mask_column="doc_attention_mask",
        doc_ids_column="doc_ids",
        k=1,
    )

    with pytest.raises(ValueError, match="row_indices must have shape"):
        retriever._materialize_output(
            row_indices=torch.zeros((1, 1), dtype=torch.long),
            scores=torch.zeros((1, 1, 1), dtype=torch.float32),
            output_device=torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="same shape as row_indices"):
        retriever._materialize_output(
            row_indices=torch.zeros((1, 1, 1), dtype=torch.long),
            scores=torch.zeros((1, 1, 2), dtype=torch.float32),
            output_device=torch.device("cpu"),
        )

    with pytest.raises(RuntimeError, match="invalid dataset row indices"):
        retriever._materialize_output(
            row_indices=-torch.ones((1, 1, 1), dtype=torch.long),
            scores=torch.zeros((1, 1, 1), dtype=torch.float32),
            output_device=torch.device("cpu"),
        )


def test_materialize_output_includes_doc_key_embeddings() -> None:
    retriever = HFDatasetRetriever(
        dataset=_build_indexed_dataset(include_doc_key_embeddings=True),
        index_name="retrieval",
        doc_tokens_column="doc_tokens",
        doc_attention_mask_column="doc_attention_mask",
        doc_ids_column="doc_ids",
        doc_key_embeddings_column="doc_key_embeddings",
        k=1,
    )

    out = retriever._materialize_output(
        row_indices=torch.zeros((1, 1, 1), dtype=torch.long),
        scores=torch.zeros((1, 1, 1), dtype=torch.float32),
        output_device=torch.device("cpu"),
    )

    assert out.doc_key_embeddings is not None
    assert tuple(out.doc_key_embeddings.shape) == (1, 1, 1, 2)


def test_validate_dataset_rejects_missing_optional_columns() -> None:
    with pytest.raises(ValueError, match="missing optional columns"):
        HFDatasetRetriever(
            dataset=_build_indexed_dataset(include_doc_ids=False, include_doc_key_embeddings=False),
            index_name="retrieval",
            doc_tokens_column="doc_tokens",
            doc_attention_mask_column="doc_attention_mask",
            doc_ids_column="doc_ids",
            doc_key_embeddings_column="doc_key_embeddings",
            k=1,
        )


def test_validate_dataset_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be between 1 and the number of dataset rows"):
        HFDatasetRetriever(
            dataset=_build_indexed_dataset(),
            index_name="retrieval",
            doc_tokens_column="doc_tokens",
            doc_attention_mask_column="doc_attention_mask",
            doc_ids_column="doc_ids",
            k=4,
        )


def test_validate_dataset_rejects_missing_required_columns() -> None:
    dataset = Dataset.from_dict(
        {
            "doc_attention_mask": [[1, 1], [1, 1], [1, 0]],
            "index_embeddings": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "doc_ids": [7, 8, 9],
        }
    )
    dataset.add_faiss_index(column="index_embeddings", index_name="retrieval")

    with pytest.raises(ValueError, match="missing required columns"):
        HFDatasetRetriever(
            dataset=dataset,
            index_name="retrieval",
            doc_tokens_column="doc_tokens",
            doc_attention_mask_column="doc_attention_mask",
            doc_ids_column="doc_ids",
            k=1,
        )


def test_validate_dataset_rejects_missing_index() -> None:
    with pytest.raises(ValueError, match="does not have a FAISS index"):
        HFDatasetRetriever(
            dataset=_build_indexed_dataset(index_name="retrieval"),
            index_name="other",
            doc_tokens_column="doc_tokens",
            doc_attention_mask_column="doc_attention_mask",
            doc_ids_column="doc_ids",
            k=1,
        )


def test_retrieve_rejects_wrong_query_rank() -> None:
    retriever = HFDatasetRetriever(
        dataset=_build_indexed_dataset(),
        index_name="retrieval",
        doc_tokens_column="doc_tokens",
        doc_attention_mask_column="doc_attention_mask",
        doc_ids_column="doc_ids",
        k=1,
    )

    with pytest.raises(ValueError, match="query_embeddings must have shape"):
        retriever.retrieve(torch.ones((1, 2), dtype=torch.float32))
