"""Test set-up and fixtures code."""

import os
import sys
from types import SimpleNamespace

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import torch
from datasets import load_from_disk
from meds_torchdata import MEDSTorchBatch
from omegaconf import OmegaConf
from torch import nn

from medrap.types import ModelOutput


def make_supervised_batch() -> MEDSTorchBatch:
    """Return a tiny labeled MEDS batch for doctests and trainer smoke tests."""
    return MEDSTorchBatch(
        code=torch.LongTensor([[1, 2, 3], [3, 2, 1]]),
        numeric_value=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numeric_value_mask=torch.BoolTensor([[False, False, False], [False, False, False]]),
        time_delta_days=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        boolean_value=torch.BoolTensor([True, False]),
    )


def make_meds_batch(batch_size: int = 4, seq_len: int = 128, vocab_size: int = 6620) -> MEDSTorchBatch:
    """Return a synthetic ``MEDSTorchBatch`` with the same structure as a real MIMIC batch.

    Shapes, dtypes, and value ranges match what ``MEDSDatamodule`` produces from
    a tensorized MIMIC-IV cohort:

    - ``code``: random integers in ``[1, vocab_size)``, shape ``(B, S)``
    - ``numeric_value``: mostly-zero floats with occasional lab values, shape ``(B, S)``
    - ``numeric_value_mask``: bool mask indicating which values are present, shape ``(B, S)``
    - ``time_delta_days``: non-negative floats (inter-event gaps in days), shape ``(B, S)``
    - ``boolean_value``: binary labels, shape ``(B,)``
    """
    numeric_value_mask = torch.rand(batch_size, seq_len) < 0.15
    numeric_value = torch.where(
        numeric_value_mask, torch.randn(batch_size, seq_len).abs() * 50, torch.zeros(batch_size, seq_len)
    )
    return MEDSTorchBatch(
        code=torch.randint(1, vocab_size, (batch_size, seq_len)),
        numeric_value=numeric_value,
        numeric_value_mask=numeric_value_mask,
        time_delta_days=torch.rand(batch_size, seq_len) * 2.0,
        boolean_value=torch.randint(0, 2, (batch_size,)).bool(),
    )


def batch_dataset(batch: MEDSTorchBatch):
    """Wrap a ``MEDSTorchBatch`` as a minimal dataset compatible with ``DataLoader``.

    Returns an object with ``__len__``, ``__getitem__``, and ``collate`` so it
    can be passed directly to ``MEDSDatamodule._train_ds`` (and siblings) in
    doctests without requiring real tensorized data files.
    """

    class _D:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, i: int) -> MEDSTorchBatch:
            return batch

        def collate(self, items: list[MEDSTorchBatch]) -> MEDSTorchBatch:
            return items[0]

    return _D()


class ModelOutputBinaryModel(nn.Module):
    """Tiny binary model returning a ``ModelOutput``."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(3)
        self.linear = nn.Linear(3, 1)

    def forward(self, batch: MEDSTorchBatch) -> ModelOutput:
        assert batch.code is not None, "This model expects a 'code' tensor in the batch."
        features = self.layer_norm(batch.code.float())
        return ModelOutput(logits=self.linear(features))


class DoctestTokenizer:
    """Tiny tokenizer stub for retrieval-preparation doctests."""

    def __call__(
        self,
        texts: list[str],
        *,
        truncation: bool,
        padding: str,
        max_length: int,
    ) -> dict[str, list[list[int]]]:
        return {
            "input_ids": [[idx + 1] * max_length for idx, _text in enumerate(texts)],
            "attention_mask": [[1] * max_length for _text in texts],
        }


class DoctestEmbedder:
    """Tiny embedder stub for retrieval-preparation doctests."""

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        rows = []
        for text in texts:
            rows.append([1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0])
        return rows


@pytest.fixture(scope="session", autouse=True)
def _setup_doctest_namespace(
    doctest_namespace: dict[str, Any],
    # You can pass more fixtures here to add them to the namespace
):
    doctest_namespace.update(
        {
            "datetime": datetime,
            "tempfile": tempfile,
            "Path": Path,
            "SimpleNamespace": SimpleNamespace,
            "OmegaConf": OmegaConf,
            "torch": torch,
            "load_from_disk": load_from_disk,
            "MEDSTorchBatch": MEDSTorchBatch,
            "make_supervised_batch": make_supervised_batch,
            "make_meds_batch": make_meds_batch,
            "batch_dataset": batch_dataset,
            "ModelOutput": ModelOutput,
            "ModelOutputBinaryModel": ModelOutputBinaryModel,
            "DoctestTokenizer": DoctestTokenizer,
            "DoctestEmbedder": DoctestEmbedder,
        }
    )
