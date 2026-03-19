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
            "MEDSTorchBatch": MEDSTorchBatch,
            "make_supervised_batch": make_supervised_batch,
            "ModelOutput": ModelOutput,
            "ModelOutputBinaryModel": ModelOutputBinaryModel,
        }
    )
