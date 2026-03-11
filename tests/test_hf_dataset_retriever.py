from dataclasses import dataclass

import pytest
import torch
from datasets import Dataset

from medrap.retrievers import (
    HFDatasetRetriever,
    HFDatasetSnapshotBuilder,
    _DatasetSnapshot,
)


class _CompletedFuture:
    def __init__(self, result: _DatasetSnapshot) -> None:
        self._result = result

    def done(self) -> bool:
        return True

    def result(self) -> _DatasetSnapshot:
        return self._result


class _FailedFuture:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def done(self) -> bool:
        return True

    def result(self) -> _DatasetSnapshot:
        raise self._exc


@dataclass
class _StubSnapshotBuilder:
    snapshot: _DatasetSnapshot

    def build_snapshot(self, refresh_context=None) -> _DatasetSnapshot:
        return self.snapshot


def _build_fake_snapshot(
    *,
    doc_tokens: list[list[int]],
    doc_ids: list[int],
    index_name: str = "retrieval",
) -> _DatasetSnapshot:
    class _FakeIndexedDataset:
        def __init__(self) -> None:
            self.column_names = ["doc_tokens", "doc_attention_mask", "doc_ids"]
            self._doc_tokens = doc_tokens
            self._doc_attention_mask = [[True] * len(tokens) for tokens in doc_tokens]
            self._doc_ids = doc_ids
            self._index_name = index_name

        def __len__(self) -> int:
            return len(self._doc_ids)

        def get_index(self, requested_index_name: str):
            if requested_index_name != self._index_name:
                raise KeyError(requested_index_name)
            return object()

        def search_batch(self, requested_index_name: str, queries, k: int):
            self.get_index(requested_index_name)
            return [[1.0] * k for _ in range(len(queries))], [[0] * k for _ in range(len(queries))]

        def __getitem__(self, row_indices):
            return {
                "doc_tokens": [self._doc_tokens[i] for i in row_indices],
                "doc_attention_mask": [self._doc_attention_mask[i] for i in row_indices],
                "doc_ids": [self._doc_ids[i] for i in row_indices],
            }

    return _DatasetSnapshot(dataset=_FakeIndexedDataset(), index_name=index_name)


def test_build_snapshot_rebuilds_key_column(monkeypatch: pytest.MonkeyPatch) -> None:
    source_dataset = Dataset.from_dict(
        {
            "doc_tokens": [[10, 11], [20, 21]],
            "doc_attention_mask": [[1, 1], [1, 0]],
            "doc_key_embeddings": [[9.0, 9.0], [9.0, 9.0]],
        }
    )

    attached_indexes: list[str] = []

    def fake_add_faiss_index(self: Dataset, *, column: str, index_name: str):
        attached_indexes.append(f"{column}:{index_name}")
        return self

    monkeypatch.setattr(Dataset, "add_faiss_index", fake_add_faiss_index)

    builder = HFDatasetSnapshotBuilder(
        source_dataset=source_dataset,
        index_name="retrieval",
        key_embedding_column="doc_key_embeddings",
        encode_documents=lambda batch, _refresh_context: [[1.0, 0.0] for _ in batch["doc_tokens"]],
        batch_size=1,
    )

    snapshot = builder.build_snapshot()

    assert snapshot.index_name == "retrieval"
    assert snapshot.dataset["doc_key_embeddings"] == [[1.0, 0.0], [1.0, 0.0]]
    assert source_dataset["doc_key_embeddings"] == [[9.0, 9.0], [9.0, 9.0]]
    assert attached_indexes == ["doc_key_embeddings:retrieval"]


def test_build_snapshot_validates_embedding_count() -> None:
    source_dataset = Dataset.from_dict(
        {
            "doc_tokens": [[10, 11], [20, 21]],
            "doc_attention_mask": [[1, 1], [1, 0]],
        }
    )

    builder = HFDatasetSnapshotBuilder(
        source_dataset=source_dataset,
        index_name="retrieval",
        key_embedding_column="doc_key_embeddings",
        encode_documents=lambda batch, _refresh_context: [[1.0, 0.0]],
        batch_size=2,
    )

    with pytest.raises(ValueError, match="one key embedding per dataset row"):
        builder.build_snapshot()


def test_poll_refresh_failure_clears_refresh_job() -> None:
    retriever = HFDatasetRetriever(
        active_snapshot=_build_fake_snapshot(doc_tokens=[[10, 11]], doc_ids=[7]),
        snapshot_builder=None,
        doc_tokens_column="doc_tokens",
        doc_attention_mask_column="doc_attention_mask",
        doc_ids_column="doc_ids",
        k=1,
    )

    retriever._refresh_job = _FailedFuture(RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        retriever._poll_refresh()

    assert retriever._refresh_job is None
    retriever.close()


def test_retrieve_swaps_in_completed_refresh_snapshot() -> None:
    initial_snapshot = _build_fake_snapshot(doc_tokens=[[10, 11]], doc_ids=[7])
    refreshed_snapshot = _build_fake_snapshot(doc_tokens=[[20, 21]], doc_ids=[8])

    retriever = HFDatasetRetriever(
        active_snapshot=initial_snapshot,
        snapshot_builder=_StubSnapshotBuilder(snapshot=refreshed_snapshot),
        doc_tokens_column="doc_tokens",
        doc_attention_mask_column="doc_attention_mask",
        doc_ids_column="doc_ids",
        k=1,
    )
    retriever._refresh_job = _CompletedFuture(refreshed_snapshot)

    out = retriever.retrieve(torch.ones((1, 1, 4), dtype=torch.float32))

    assert out.doc_ids.tolist() == [[[8]]]
    assert retriever._active_snapshot is refreshed_snapshot
    retriever.close()


def test_start_refresh_returns_false_when_job_is_running() -> None:
    snapshot = _build_fake_snapshot(doc_tokens=[[10, 11]], doc_ids=[7])
    retriever = HFDatasetRetriever(
        active_snapshot=snapshot,
        snapshot_builder=_StubSnapshotBuilder(snapshot=snapshot),
        doc_tokens_column="doc_tokens",
        doc_attention_mask_column="doc_attention_mask",
        doc_ids_column="doc_ids",
        k=1,
    )
    retriever._refresh_job = _CompletedFuture(snapshot)

    assert retriever.start_refresh() is False
    retriever.close()
