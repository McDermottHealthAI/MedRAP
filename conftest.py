"""Test set-up and fixtures code."""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import torch
from meds_torchdata import MEDSTorchBatch

from medrap.retrievers import _DatasetSnapshot


class FakeIndexedDataset:
    def __init__(
        self,
        *,
        doc_tokens,
        doc_attention_mask,
        doc_ids,
        index_name: str = "embeddings",
    ) -> None:
        self.column_names = ["doc_tokens", "doc_attention_mask", "doc_ids"]
        self._doc_tokens = doc_tokens
        self._doc_attention_mask = doc_attention_mask
        self._doc_ids = doc_ids
        self._index_name = index_name

    def __len__(self) -> int:
        return len(self._doc_ids)

    def get_index(self, index_name: str):
        if index_name != self._index_name:
            raise KeyError(index_name)
        return object()

    def search_batch(self, index_name: str, queries, k: int):
        self.get_index(index_name)
        n_queries = len(queries)
        return [[1.0] * k for _ in range(n_queries)], [[0] * k for _ in range(n_queries)]

    def __getitem__(self, row_indices):
        return {
            "doc_tokens": [self._doc_tokens[i] for i in row_indices],
            "doc_attention_mask": [self._doc_attention_mask[i] for i in row_indices],
            "doc_ids": [self._doc_ids[i] for i in row_indices],
        }


class FakeBuildDataset:
    def __init__(
        self,
        *,
        columns: dict[str, list[object]],
        index_name: str | None = None,
    ) -> None:
        self._columns = columns
        self.column_names = list(columns.keys())
        self._index_name = index_name

    def __len__(self) -> int:
        return len(next(iter(self._columns.values()), []))

    def __getitem__(self, column_name: str) -> list[object]:
        return self._columns[column_name]

    def remove_columns(self, column_name: str):
        columns = {name: values for name, values in self._columns.items() if name != column_name}
        return FakeBuildDataset(columns=columns, index_name=self._index_name)

    def map(self, function, *, batched: bool, batch_size: int):
        assert batched is True
        updated_columns = {name: list(values) for name, values in self._columns.items()}
        row_count = len(self)
        for start in range(0, row_count, batch_size):
            stop = min(start + batch_size, row_count)
            batch = {name: values[start:stop] for name, values in self._columns.items()}
            mapped = function(batch)
            for name, values in mapped.items():
                updated_columns.setdefault(name, [None] * row_count)
                updated_columns[name][start:stop] = values
        return FakeBuildDataset(columns=updated_columns, index_name=self._index_name)

    def add_faiss_index(self, *, column: str, index_name: str):
        if column not in self._columns:
            raise KeyError(column)
        self._index_name = index_name
        return self


@pytest.fixture(scope="session", autouse=True)
def _setup_doctest_namespace(
    doctest_namespace: dict[str, Any],
    # You can pass more fixtures here to add them to the namespace
):
    doctest_namespace.update(
        {
            "datetime": datetime,
            "FakeBuildDataset": FakeBuildDataset,
            "FakeIndexedDataset": FakeIndexedDataset,
            "tempfile": tempfile,
            "Path": Path,
            "_DatasetSnapshot": _DatasetSnapshot,
            "torch": torch,
            "MEDSTorchBatch": MEDSTorchBatch,
        }
    )
