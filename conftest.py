"""Test set-up and fixtures code."""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import torch
from meds_torchdata import MEDSTorchBatch


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
        }
    )
