"""Test set-up and fixtures code."""

import os
import sys

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
            "torch": torch,
            "load_from_disk": load_from_disk,
            "MEDSTorchBatch": MEDSTorchBatch,
            "make_supervised_batch": make_supervised_batch,
            "ModelOutput": ModelOutput,
            "ModelOutputBinaryModel": ModelOutputBinaryModel,
            "DoctestTokenizer": DoctestTokenizer,
            "DoctestEmbedder": DoctestEmbedder,
        }
    )
