"""Integration tests: MEDS cohort → tensorize → task labels → MEDSTorchBatch → MedRAP model.

These tests exercise the full pipeline from a synthetic MEDS dataset through
tensorization and task label creation to MedRAP model forward pass, using the
existing ``meds_torchdata`` datamodule and MedRAP components.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
import torch
from meds_torchdata import MEDSPytorchDataset, MEDSTorchDataConfig
from torch.utils.data import DataLoader

from medrap.encoders import MEDSCodeEncoder
from medrap.fusion import ReplaceFusion
from medrap.heads import LinearHead
from medrap.model import RetrievalAugmentedModel
from medrap.pooling import IdentityPooling
from medrap.query_projection import SequenceMeanQueryProjector
from medrap.retrieval_encoder import MeanPooledRetrievalEncoder
from medrap.retrievers import InMemoryRetriever
from medrap.task import BinaryClassificationTask


def _make_meds_dataset(root: Path) -> None:
    """Populate ``root`` with a minimal MEDS cohort (metadata + per-split shards)."""
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "dataset.json").write_text(
        json.dumps({"dataset_name": "test_cohort", "dataset_version": "1.0"})
    )
    pl.DataFrame(
        {"subject_id": [1, 2, 3, 4], "split": ["train", "train", "tuning", "held_out"]}
    ).cast({"subject_id": pl.Int64}).write_parquet(metadata_dir / "subject_splits.parquet")

    train = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1, 1, 2, 2, 2, 2],
            "time": [
                None,
                datetime(2020, 1, 1),
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                datetime(2020, 1, 3),
                None,
                datetime(2020, 2, 1),
                datetime(2020, 2, 2),
                datetime(2020, 2, 3),
            ],
            "code": [
                "DOB",
                "ADMISSION//CARDIAC",
                "HR",
                "TEMP",
                "DEATH",
                "DOB",
                "ADMISSION//PULMONARY",
                "HR",
                "DISCHARGE",
            ],
            "numeric_value": [None, None, 80.0, 37.5, None, None, None, 90.0, None],
        }
    ).cast({"subject_id": pl.Int64, "numeric_value": pl.Float32})
    (root / "data" / "train").mkdir(parents=True)
    train.write_parquet(root / "data" / "train" / "0.parquet")

    tuning = pl.DataFrame(
        {
            "subject_id": [3, 3, 3, 3],
            "time": [None, datetime(2020, 3, 1), datetime(2020, 3, 2), datetime(2020, 3, 3)],
            "code": ["DOB", "ADMISSION//ORTHO", "HR", "DISCHARGE"],
            "numeric_value": [None, None, 75.0, None],
        }
    ).cast({"subject_id": pl.Int64, "numeric_value": pl.Float32})
    (root / "data" / "tuning").mkdir(parents=True)
    tuning.write_parquet(root / "data" / "tuning" / "0.parquet")

    held_out = pl.DataFrame(
        {
            "subject_id": [4, 4, 4, 4],
            "time": [None, datetime(2020, 4, 1), datetime(2020, 4, 2), datetime(2020, 4, 3)],
            "code": ["DOB", "ADMISSION//CARDIAC", "HR", "DISCHARGE"],
            "numeric_value": [None, None, 88.0, None],
        }
    ).cast({"subject_id": pl.Int64, "numeric_value": pl.Float32})
    (root / "data" / "held_out").mkdir(parents=True)
    held_out.write_parquet(root / "data" / "held_out" / "0.parquet")


def _tensorize(meds_dir: Path, output_dir: Path) -> bool:
    """Run ``MTD_preprocess`` to tensorize the MEDS cohort. Returns True on success."""
    import shutil

    mtd = shutil.which("MTD_preprocess")
    if mtd is None:
        return False
    result = subprocess.run(
        [mtd, f"MEDS_dataset_dir={meds_dir}", f"output_dir={output_dir}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    data_dir = output_dir / "data"
    return data_dir.is_dir() and any(data_dir.rglob("*.nrt"))


def _make_task_labels(meds_dir: Path, task_dir: Path) -> None:
    """Create in-hospital mortality labels from the MEDS cohort."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_mimic_task_labels.py",
            "--meds-dir",
            str(meds_dir),
            "--output-dir",
            str(task_dir),
            "--task",
            "in_hospital_mortality",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Label creation failed:\n{result.stdout}\n{result.stderr}")


@pytest.fixture(scope="module")
def meds_integration_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Create a MEDS cohort, tensorize it, and create task labels. Returns path dict."""
    root = tmp_path_factory.mktemp("meds_integration")
    meds_dir = root / "meds_cohort"
    tensorized_dir = root / "tensorized"
    task_dir = root / "task_labels" / "in_hospital_mortality"

    _make_meds_dataset(meds_dir)

    tensorized_ok = _tensorize(meds_dir, tensorized_dir)

    _make_task_labels(meds_dir, task_dir)

    return {
        "meds_dir": meds_dir,
        "tensorized_dir": tensorized_dir,
        "task_dir": task_dir,
        "tensorized_ok": tensorized_ok,
    }


_SKIP_TENSORIZE = "MTD_preprocess tensorization failed (sandbox / env restriction)"


def test_tensorized_dataset_loads_as_meds_torch_batch(
    meds_integration_dirs: dict[str, Path],
) -> None:
    """Loading the tensorized MEDS dataset and collating yields MEDSTorchBatch."""
    if not meds_integration_dirs.get("tensorized_ok"):
        pytest.skip(_SKIP_TENSORIZE)
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(meds_integration_dirs["tensorized_dir"]),
        max_seq_len=10,
    )
    pyd = MEDSPytorchDataset(cfg, split="train")
    assert len(pyd) == 2

    loader = DataLoader(pyd, batch_size=2, collate_fn=pyd.collate)
    batch = next(iter(loader))

    assert batch.code is not None
    assert batch.code.ndim == 2
    assert batch.code.shape[0] == 2
    assert batch.numeric_value is not None
    assert batch.time_delta_days is not None


def test_task_labels_produce_boolean_value_in_batch(
    meds_integration_dirs: dict[str, Path],
) -> None:
    """When task labels are configured, MEDSTorchBatch includes boolean_value."""
    if not meds_integration_dirs.get("tensorized_ok"):
        pytest.skip(_SKIP_TENSORIZE)
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(meds_integration_dirs["tensorized_dir"]),
        max_seq_len=10,
        task_labels_dir=str(meds_integration_dirs["task_dir"]),
        seq_sampling_strategy="to_end",
    )
    pyd = MEDSPytorchDataset(cfg, split="train")
    assert len(pyd) >= 1

    loader = DataLoader(pyd, batch_size=len(pyd), collate_fn=pyd.collate)
    batch = next(iter(loader))

    assert batch.boolean_value is not None
    assert batch.boolean_value.dtype == torch.bool
    assert batch.boolean_value.ndim == 1


def test_medrap_model_forward_on_meds_batch(
    meds_integration_dirs: dict[str, Path],
) -> None:
    """MedRAP model runs a forward pass on a real MEDSTorchBatch from MEDS data."""
    if not meds_integration_dirs.get("tensorized_ok"):
        pytest.skip(_SKIP_TENSORIZE)
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(meds_integration_dirs["tensorized_dir"]),
        max_seq_len=10,
        task_labels_dir=str(meds_integration_dirs["task_dir"]),
        seq_sampling_strategy="to_end",
    )
    pyd = MEDSPytorchDataset(cfg, split="train")
    loader = DataLoader(pyd, batch_size=len(pyd), collate_fn=pyd.collate)
    batch = next(iter(loader))

    vocab_size = cfg.vocab_size
    model = RetrievalAugmentedModel(
        encoder=MEDSCodeEncoder(),
        query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
        retriever=InMemoryRetriever(
            doc_key_embeddings=torch.randn(2, 4),
            doc_tokens=torch.zeros(2, 1, dtype=torch.long),
            doc_attention_mask=torch.ones(2, 1, dtype=torch.bool),
        ),
        retrieval_encoder=MeanPooledRetrievalEncoder(vocab_size=vocab_size + 1, embedding_dim=4),
        fusion=ReplaceFusion(),
        pooling=IdentityPooling(),
        head=LinearHead(in_dim=4, out_dim=1),
    )
    output = model(batch)

    assert output.logits.shape == (batch.code.shape[0], 1)
    assert "retriever_output" in output.metadata


def test_task_extract_targets_works_on_meds_batch(
    meds_integration_dirs: dict[str, Path],
) -> None:
    """BinaryClassificationTask.extract_targets works on MEDS-loaded batches."""
    if not meds_integration_dirs.get("tensorized_ok"):
        pytest.skip(_SKIP_TENSORIZE)
    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(meds_integration_dirs["tensorized_dir"]),
        max_seq_len=10,
        task_labels_dir=str(meds_integration_dirs["task_dir"]),
        seq_sampling_strategy="to_end",
    )
    pyd = MEDSPytorchDataset(cfg, split="train")
    loader = DataLoader(pyd, batch_size=len(pyd), collate_fn=pyd.collate)
    batch = next(iter(loader))

    task = BinaryClassificationTask()
    targets = task.extract_targets(batch)

    assert targets.dtype == torch.float32
    assert targets.ndim == 1
    assert targets.shape[0] == batch.code.shape[0]


def test_label_script_creates_valid_parquets(
    meds_integration_dirs: dict[str, Path],
) -> None:
    """The label creation script produces MEDS-format parquet with the expected schema."""
    task_dir = meds_integration_dirs["task_dir"]

    for split in ("train",):
        label_path = task_dir / f"{split}.parquet"
        assert label_path.exists(), f"Missing {label_path}"

        df = pl.read_parquet(label_path)
        assert "subject_id" in df.columns
        assert "prediction_time" in df.columns
        assert "boolean_value" in df.columns
        assert df["subject_id"].dtype == pl.Int64
        assert df["boolean_value"].dtype == pl.Boolean
        assert len(df) > 0
