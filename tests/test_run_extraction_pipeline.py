"""Unit tests for the resolver helper in ``scripts/run_extraction_pipeline.py``.

The wrapper itself is mostly subprocess wiring; only the retrieval-db resolver has logic worth covering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from run_extraction_pipeline import _resolve_retrieval_db  # noqa: E402


def _write_config(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(payload), run_dir / "config.yaml")


def test_resolve_retrieval_db_from_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_config(run_dir, {"retriever": {"dataset_path": "/foo/bar"}})

    assert _resolve_retrieval_db(run_dir, override=None) == Path("/foo/bar")


def test_resolve_retrieval_db_override_wins(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_config(run_dir, {"retriever": {"dataset_path": "/from/config"}})

    assert _resolve_retrieval_db(run_dir, override=Path("/from/cli")) == Path("/from/cli")


def test_resolve_retrieval_db_errors_when_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_config(run_dir, {"retriever": {"k": 4}})  # no dataset_path

    with pytest.raises(ValueError, match="retrieval"):
        _resolve_retrieval_db(run_dir, override=None)
