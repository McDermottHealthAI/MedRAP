"""Multi-task datamodule: augments MEDS batches with per-task binary labels."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning
import polars as pl
import torch
from meds_torchdata.extensions.lightning_datamodule import Datamodule as MEDSLightningDatamodule
from meds_torchdata.pytorch_dataset import MEDSPytorchDataset

if TYPE_CHECKING:
    from meds_torchdata import MEDSTorchBatch, MEDSTorchDataConfig
    from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

_SPLIT_FILES = {"train": "train.parquet", "tuning": "tuning.parquet", "held_out": "held_out.parquet"}

MT_LABEL_KEY = "multi_task_labels"


class MultiTaskMEDSDataset(MEDSPytorchDataset):
    """MEDSPytorchDataset subclass that injects multi-task binary labels into each item.

    Labels are loaded from a parquet file with columns::

        subject_id | prediction_time | task_0 | task_1 | ... | task_{N-1}

    For each sample, the corresponding (N,) float32 label vector is attached to the
    returned item dict as ``"multi_task_labels"``.  Missing entries (subjects not found
    in the label file) are filled with ``NaN`` so the loss can mask them.
    """

    def __init__(
        self,
        config: MEDSTorchDataConfig,
        split: str,
        mt_labels_dir: str | Path,
        num_tasks: int,
    ) -> None:
        super().__init__(config, split)
        self._num_tasks = num_tasks
        self._mt_index: dict[tuple[int, object], int] = {}
        self._mt_matrix: torch.Tensor = torch.empty(0, num_tasks, dtype=torch.float32)
        self._load_mt_labels(Path(mt_labels_dir), split)

    def _load_mt_labels(self, labels_dir: Path, split: str) -> None:
        fname = _SPLIT_FILES.get(split)
        if fname is None:
            raise ValueError(f"Unknown split {split!r}; expected one of {list(_SPLIT_FILES)}")
        label_path = labels_dir / fname
        if not label_path.exists():
            log.warning("No multi-task label file found at %s; all labels will be NaN.", label_path)
            return

        task_cols = [f"task_{i}" for i in range(self._num_tasks)]
        df = pl.read_parquet(label_path, columns=["subject_id", "prediction_time", *task_cols])

        self._mt_matrix = torch.tensor(df.select(task_cols).to_numpy(), dtype=torch.float32)
        subject_ids = df["subject_id"].to_list()
        pred_times = df["prediction_time"].to_list()
        self._mt_index = {
            (int(sid), pt): i for i, (sid, pt) in enumerate(zip(subject_ids, pred_times, strict=False))
        }
        log.info("Loaded %d multi-task label rows for split %s.", len(self._mt_index), split)

    def _seeded_getitem(self, idx: int, seed: int | None = None) -> dict[str, Any]:
        item = super()._seeded_getitem(idx, seed)

        subject_id = int(self.index[idx][0])
        pred_time = self.schema_df[idx, "prediction_time"]
        key = (subject_id, pred_time)

        row_idx = self._mt_index.get(key)
        labels = (
            self._mt_matrix[row_idx].clone()
            if row_idx is not None
            else torch.full((self._num_tasks,), float("nan"), dtype=torch.float32)
        )

        item[MT_LABEL_KEY] = labels
        return item

    def collate(self, batch: list[dict[str, Any]]) -> MEDSTorchBatch:
        mt_labels = [item.pop(MT_LABEL_KEY) for item in batch]
        meds_batch = super().collate(batch)
        meds_batch.multi_task_labels = torch.stack(mt_labels)  # (B, N)
        return meds_batch


class MultiTaskMEDSDatamodule(lightning.LightningDataModule):
    """Lightning datamodule for multi-task binary code prediction.

    Wraps ``meds_torchdata.extensions.lightning_datamodule.Datamodule`` with
    :class:`MultiTaskMEDSDataset` as the dataset class so that each batch carries
    ``batch.multi_task_labels: (B, N)`` in addition to the standard MEDS fields.

    Args:
        config: Standard :class:`meds_torchdata.MEDSTorchDataConfig`.
        mt_labels_dir: Directory containing ``train.parquet``, ``tuning.parquet``,
            ``held_out.parquet`` produced by ``scripts/prepare_multi_task_labels.py``.
        num_tasks: Number of prediction tasks (N).  Must match the number of ``task_*``
            columns in the label parquets.
        batch_size: Dataloader batch size.
        num_workers: Dataloader worker count.
        pin_memory: Whether to pin dataloader memory.
    """

    def __init__(
        self,
        config: MEDSTorchDataConfig,
        mt_labels_dir: str,
        num_tasks: int,
        batch_size: int = 32,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
    ) -> None:
        super().__init__()
        self._inner = MEDSLightningDatamodule(
            config=config,
            data_class=_make_dataset_class(mt_labels_dir=mt_labels_dir, num_tasks=num_tasks),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def setup(self, stage: str | None = None) -> None:
        self._inner.setup(stage)

    def train_dataloader(self) -> DataLoader:
        return self._inner.train_dataloader()

    def val_dataloader(self) -> DataLoader:
        return self._inner.val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return self._inner.test_dataloader()

    @property
    def train_dataset(self):
        """Forward to the inner ``MEDSLightningDatamodule.train_dataset``."""
        return self._inner.train_dataset

    @property
    def val_dataset(self):
        """Forward to the inner ``MEDSLightningDatamodule.val_dataset``.

        Required by ``extract_val_schema`` (used by the demographic-heatmap and
        llm-judge pipelines) to recover ``(subject_id, end_event_index,
        prediction_time)`` for each row of the val split.
        """
        return self._inner.val_dataset

    @property
    def test_dataset(self):
        """Forward to the inner ``MEDSLightningDatamodule.test_dataset``."""
        return self._inner.test_dataset


def _make_dataset_class(*, mt_labels_dir: str, num_tasks: int) -> type[MultiTaskMEDSDataset]:
    """Return a MultiTaskMEDSDataset subclass with mt_labels_dir and num_tasks baked in.

    The ``Datamodule`` ``data_class`` parameter expects a class whose constructor signature
    matches ``MEDSPytorchDataset(config, split)``.  We partially apply the extra arguments
    via a thin subclass.
    """

    class _BoundDataset(MultiTaskMEDSDataset):
        def __init__(self, config: MEDSTorchDataConfig, split: str) -> None:
            super().__init__(config, split, mt_labels_dir=mt_labels_dir, num_tasks=num_tasks)

    _BoundDataset.__name__ = "MultiTaskMEDSDataset"
    _BoundDataset.__qualname__ = "MultiTaskMEDSDataset"
    return _BoundDataset


def load_code_index(mt_labels_dir: str | Path) -> dict[int, str]:
    """Load the task-index → code-string mapping saved by the label prep script."""
    path = Path(mt_labels_dir) / "code_index.json"
    raw: dict[str, str] = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}
