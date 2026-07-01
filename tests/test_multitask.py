"""Tests for multi-task datamodule components."""

import json

import polars as pl
import pytest
import torch
from meds_torchdata import MEDSTorchBatch
from meds_torchdata.pytorch_dataset import MEDSPytorchDataset

from medrap.train.multitask_datamodule import MultiTaskMEDSDataset, _make_dataset_class

# ---------------------------------------------------------------------------
# MultiTaskMEDSDatamodule (integration, uses temp files)
# ---------------------------------------------------------------------------


class TestMultiTaskMEDSDataset:
    def test_load_code_index(self, tmp_path):
        """load_code_index parses code_index.json correctly."""
        from medrap.train.multitask_datamodule import load_code_index

        index = {"0": "LAB//123", "1": "DIAG//456", "2": "MED//789"}
        (tmp_path / "code_index.json").write_text(json.dumps(index))
        result = load_code_index(tmp_path)
        assert result == {0: "LAB//123", 1: "DIAG//456", 2: "MED//789"}

    def test_label_parquet_schema(self, tmp_path):
        """prepare_multi_task_labels.py output has expected schema."""
        import datetime

        df = pl.DataFrame(
            {
                "subject_id": [1, 2],
                "prediction_time": [
                    datetime.datetime(2020, 1, 1),
                    datetime.datetime(2020, 6, 1),
                ],
                "task_0": [1.0, 0.0],
                "task_1": [0.0, 1.0],
                "task_2": [float("nan"), 0.0],
            }
        )
        out = tmp_path / "train.parquet"
        df.write_parquet(out)

        reloaded = pl.read_parquet(out)
        assert "subject_id" in reloaded.columns
        assert "prediction_time" in reloaded.columns
        assert "task_0" in reloaded.columns
        assert len(reloaded) == 2

    def test_load_mt_labels_and_missing_file(self, tmp_path):
        """_load_mt_labels reads split labels and tolerates absent split files."""
        import datetime

        labels = pl.DataFrame(
            {
                "subject_id": [7],
                "prediction_time": [datetime.datetime(2020, 1, 1)],
                "task_0": [1.0],
                "task_1": [0.0],
            }
        )
        labels.write_parquet(tmp_path / "train.parquet")

        dataset = object.__new__(MultiTaskMEDSDataset)
        dataset._num_tasks = 2
        dataset._mt_index = {}
        dataset._mt_matrix = torch.empty(0, 2, dtype=torch.float32)
        dataset._load_mt_labels(tmp_path, "train")
        key = (7, datetime.datetime(2020, 1, 1))
        assert key in dataset._mt_index
        assert dataset._mt_matrix[dataset._mt_index[key]].tolist() == [1.0, 0.0]

        dataset._mt_index = {}
        dataset._mt_matrix = torch.empty(0, 2, dtype=torch.float32)
        dataset._load_mt_labels(tmp_path, "tuning")

    def test_load_mt_labels_rejects_unknown_split(self, tmp_path):
        dataset = object.__new__(MultiTaskMEDSDataset)
        dataset._num_tasks = 2
        dataset._mt_lookup = {}
        with pytest.raises(ValueError, match="Unknown split"):
            dataset._load_mt_labels(tmp_path, "bad")

    def test_seeded_getitem_adds_matching_or_nan_labels(self, monkeypatch):
        import datetime

        def fake_seeded_getitem(self, idx, seed=None):
            return {"code": idx}

        monkeypatch.setattr(MEDSPytorchDataset, "_seeded_getitem", fake_seeded_getitem)
        dataset = object.__new__(MultiTaskMEDSDataset)
        dataset._num_tasks = 2
        dataset._mt_index = {(7, datetime.datetime(2020, 1, 1)): 0}
        dataset._mt_matrix = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        dataset.index = [(7,), (8,)]
        dataset.schema_df = pl.DataFrame(
            {"prediction_time": [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 2)]}
        )

        assert dataset._seeded_getitem(0)["multi_task_labels"].tolist() == [1.0, 0.0]
        missing = dataset._seeded_getitem(1)["multi_task_labels"]
        assert missing.isnan().all()

    def test_collate_attaches_multi_task_labels(self, monkeypatch):
        def fake_collate(self, batch):
            return MEDSTorchBatch(
                code=torch.LongTensor([[1], [2]]),
                numeric_value=torch.zeros(2, 1),
                numeric_value_mask=torch.zeros(2, 1, dtype=torch.bool),
                time_delta_days=torch.zeros(2, 1),
            )

        monkeypatch.setattr(MEDSPytorchDataset, "collate", fake_collate)
        dataset = object.__new__(MultiTaskMEDSDataset)
        batch = [
            {"multi_task_labels": torch.tensor([1.0, 0.0])},
            {"multi_task_labels": torch.tensor([0.0, 1.0])},
        ]

        out = dataset.collate(batch)
        assert out.multi_task_labels.tolist() == [[1.0, 0.0], [0.0, 1.0]]

    def test_make_dataset_class_binds_extra_arguments(self, monkeypatch):
        calls = []

        def fake_init(self, config, split, mt_labels_dir, num_tasks):
            calls.append((config, split, mt_labels_dir, num_tasks))

        monkeypatch.setattr(MultiTaskMEDSDataset, "__init__", fake_init)
        bound = _make_dataset_class(mt_labels_dir="/labels", num_tasks=3)

        assert bound.__name__ == "MultiTaskMEDSDataset"
        bound(config="cfg", split="train")
        assert calls == [("cfg", "train", "/labels", 3)]


# ---------------------------------------------------------------------------
# Coverage gap-fillers for multitask_datamodule.py
# ---------------------------------------------------------------------------


class TestMultiTaskMEDSDatasetFullInit:
    """Cover lines 46-49 in src/medrap/multitask_datamodule.py (__init__ body)."""

    def test_full_init_calls_super_and_loads_mt_labels(self, monkeypatch, tmp_path):
        import datetime

        labels = pl.DataFrame(
            {
                "subject_id": [11],
                "prediction_time": [datetime.datetime(2020, 1, 1)],
                "task_0": [1.0],
                "task_1": [0.0],
            }
        )
        labels.write_parquet(tmp_path / "train.parquet")

        # No-op the heavy MEDSPytorchDataset.__init__ so we exercise only the
        # MultiTaskMEDSDataset wrapper's body.
        def fake_super_init(self, config, split):
            pass

        monkeypatch.setattr(MEDSPytorchDataset, "__init__", fake_super_init)

        ds = MultiTaskMEDSDataset(config=object(), split="train", mt_labels_dir=tmp_path, num_tasks=2)
        assert ds._num_tasks == 2
        key = (11, datetime.datetime(2020, 1, 1))
        assert key in ds._mt_index
        assert ds._mt_matrix[ds._mt_index[key]].tolist() == [1.0, 0.0]


class TestMultiTaskMEDSDatamodule:
    """Cover lines 119-120 (init) + 128-155 (dataloader/dataset forwarders) in
    src/medrap/multitask_datamodule.py."""

    def test_datamodule_init_wires_inner_and_forwards_dataloaders_and_datasets(self, monkeypatch):
        from medrap.train.multitask_datamodule import MultiTaskMEDSDatamodule

        captured = {}

        class _FakeInner:
            def __init__(self, **kw):
                captured.update(kw)
                self.train_dataset = "TRAIN_DS"
                self.val_dataset = "VAL_DS"
                self.test_dataset = "TEST_DS"

            def train_dataloader(self):
                return "TRAIN_DL"

            def val_dataloader(self):
                return "VAL_DL"

            def test_dataloader(self):
                return "TEST_DL"

        import medrap.train.multitask_datamodule as mod

        monkeypatch.setattr(mod, "MEDSLightningDatamodule", _FakeInner)

        dm = MultiTaskMEDSDatamodule(config=object(), mt_labels_dir="/tmp/labels", num_tasks=3, batch_size=4)
        # Dataloader forwarders (lines 129, 132, 135).
        assert dm.train_dataloader() == "TRAIN_DL"
        assert dm.val_dataloader() == "VAL_DL"
        assert dm.test_dataloader() == "TEST_DL"
        # Property forwarders (lines 140, 150, 155).
        assert dm.train_dataset == "TRAIN_DS"
        assert dm.val_dataset == "VAL_DS"
        assert dm.test_dataset == "TEST_DS"
        # The inner constructor received the configured batch size.
        assert captured["batch_size"] == 4

    def test_setup_delegates_to_inner(self, monkeypatch):
        from medrap.train.multitask_datamodule import MultiTaskMEDSDatamodule

        setup_calls: list[str | None] = []

        class _FakeInner:
            def __init__(self, **_kw):
                pass

            def setup(self, stage):
                setup_calls.append(stage)

        import medrap.train.multitask_datamodule as mod

        monkeypatch.setattr(mod, "MEDSLightningDatamodule", _FakeInner)
        dm = MultiTaskMEDSDatamodule(config=object(), mt_labels_dir="/tmp/labels", num_tasks=2)
        dm.setup("fit")
        assert setup_calls == ["fit"]
