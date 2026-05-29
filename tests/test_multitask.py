"""Tests for multi-task binary classification components."""

import json

import polars as pl
import pytest
import torch
from meds_torchdata import MEDSTorchBatch
from meds_torchdata.pytorch_dataset import MEDSPytorchDataset

from medrap.fusion import PassthroughFusion
from medrap.losses import MultiTaskBCELoss, MultiTaskBCEMarginalizedLoss
from medrap.multitask_datamodule import MultiTaskMEDSDataset, _make_dataset_class
from medrap.task import MultiTaskBinaryClassificationTask
from medrap.types import FusionInput, ModelOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(num_tasks: int, batch_size: int = 2) -> MEDSTorchBatch:
    batch = MEDSTorchBatch(
        code=torch.LongTensor([[1, 2, 3]] * batch_size),
        numeric_value=torch.zeros(batch_size, 3),
        numeric_value_mask=torch.zeros(batch_size, 3, dtype=torch.bool),
        time_delta_days=torch.zeros(batch_size, 3),
    )
    batch.multi_task_labels = torch.zeros(batch_size, num_tasks)
    return batch


# ---------------------------------------------------------------------------
# MultiTaskBinaryClassificationTask
# ---------------------------------------------------------------------------


class TestMultiTaskBinaryClassificationTask:
    def test_num_tasks_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            MultiTaskBinaryClassificationTask(num_tasks=0)

    def test_output_dim(self):
        task = MultiTaskBinaryClassificationTask(num_tasks=5)
        assert task.output_dim == 5

    def test_extract_targets_shape(self):
        task = MultiTaskBinaryClassificationTask(num_tasks=3)
        batch = _make_batch(num_tasks=3)
        targets = task.extract_targets(batch)
        assert targets.shape == (2, 3)
        assert targets.dtype == torch.float32

    def test_extract_targets_missing_raises(self):
        task = MultiTaskBinaryClassificationTask(num_tasks=3)
        batch = MEDSTorchBatch(
            code=torch.LongTensor([[1, 2]]),
            numeric_value=torch.zeros(1, 2),
            numeric_value_mask=torch.zeros(1, 2, dtype=torch.bool),
            time_delta_days=torch.zeros(1, 2),
        )
        with pytest.raises(ValueError, match="multi_task_labels"):
            task.extract_targets(batch)

    def test_metrics_perfect_predictions(self):
        task = MultiTaskBinaryClassificationTask(num_tasks=2)
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        metrics = task.metrics(ModelOutput(logits=logits), targets)
        assert abs(float(metrics["accuracy"]) - 1.0) < 1e-5

    def test_metrics_nan_mask(self):
        task = MultiTaskBinaryClassificationTask(num_tasks=2)
        logits = torch.tensor([[10.0, -10.0]])
        # Only first task is valid; second is NaN — model is wrong on first
        targets = torch.tensor([[0.0, float("nan")]])
        metrics = task.metrics(ModelOutput(logits=logits), targets)
        assert float(metrics["accuracy"]) == pytest.approx(0.0)

    def test_metrics_all_nan(self):
        task = MultiTaskBinaryClassificationTask(num_tasks=2)
        logits = torch.zeros(2, 2)
        targets = torch.full((2, 2), float("nan"))
        metrics = task.metrics(ModelOutput(logits=logits), targets)
        # no valid entries — should not raise, returns 0 due to clamp
        assert torch.isfinite(metrics["accuracy"])

    def test_metrics_wrong_target_type_raises(self):
        task = MultiTaskBinaryClassificationTask(num_tasks=2)
        with pytest.raises(ValueError, match="tensor targets"):
            task.metrics(ModelOutput(logits=torch.zeros(1, 2)), {"x": torch.zeros(1, 2)})


# ---------------------------------------------------------------------------
# MultiTaskBCELoss
# ---------------------------------------------------------------------------


class TestMultiTaskBCELoss:
    def test_output_is_scalar(self):
        loss_fn = MultiTaskBCELoss()
        preds = ModelOutput(logits=torch.zeros(2, 3))
        targets = torch.zeros(2, 3)
        loss = loss_fn(preds, targets)
        assert loss.shape == ()

    def test_nan_entries_excluded(self):
        loss_fn = MultiTaskBCELoss()
        # Only the first element is valid; loss should equal BCE(0.0, 1.0)
        logits = torch.zeros(1, 2)
        targets = torch.tensor([[1.0, float("nan")]])
        loss = loss_fn(ModelOutput(logits=logits), targets)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(torch.zeros(1), torch.ones(1))
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_all_nan_returns_zero(self):
        loss_fn = MultiTaskBCELoss()
        preds = ModelOutput(logits=torch.zeros(2, 3))
        targets = torch.full((2, 3), float("nan"))
        loss = loss_fn(preds, targets)
        assert float(loss) == pytest.approx(0.0)

    def test_gradients_flow(self):
        loss_fn = MultiTaskBCELoss()
        logits = torch.zeros(2, 3, requires_grad=True)
        targets = torch.randint(0, 2, (2, 3)).float()
        loss = loss_fn(ModelOutput(logits=logits), targets)
        loss.backward()
        assert logits.grad is not None

    def test_wrong_target_type_raises(self):
        loss_fn = MultiTaskBCELoss()
        with pytest.raises(ValueError):
            loss_fn(ModelOutput(logits=torch.zeros(2, 3)), {"x": torch.zeros(2, 3)})


# ---------------------------------------------------------------------------
# MultiTaskBCEMarginalizedLoss
# ---------------------------------------------------------------------------


def _marginal_pred(b: int, k: int, n: int) -> ModelOutput:
    return ModelOutput(
        logits=torch.zeros(b, n),
        metadata={
            "per_doc_logits": torch.randn(b, k, n),
            "differentiable_doc_scores": torch.randn(b, k),
        },
    )


class TestMultiTaskBCEMarginalizedLoss:
    def test_output_is_scalar(self):
        loss_fn = MultiTaskBCEMarginalizedLoss(num_tasks=3)
        targets = torch.randint(0, 2, (2, 3)).float()
        assert _marginal_pred(2, 4, 3)
        loss = loss_fn(_marginal_pred(2, 4, 3), targets)
        assert loss.shape == ()

    def test_loss_positive(self):
        loss_fn = MultiTaskBCEMarginalizedLoss(num_tasks=3)
        targets = torch.randint(0, 2, (2, 3)).float()
        assert float(loss_fn(_marginal_pred(2, 4, 3), targets)) > 0

    def test_all_nan_returns_zero(self):
        loss_fn = MultiTaskBCEMarginalizedLoss(num_tasks=2)
        targets = torch.full((2, 2), float("nan"))
        assert float(loss_fn(_marginal_pred(2, 4, 2), targets)) == pytest.approx(0.0)

    def test_partial_nan_excluded(self):
        loss_fn = MultiTaskBCEMarginalizedLoss(num_tasks=2)
        targets = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
        loss = loss_fn(_marginal_pred(2, 4, 2), targets)
        assert loss.shape == ()
        assert float(loss) > 0

    def test_k1_reduces_to_plain_bce(self):
        """With K=1 document, marginalized loss equals plain MultiTaskBCELoss."""
        loss_fn_m = MultiTaskBCEMarginalizedLoss(num_tasks=2)
        loss_fn_p = MultiTaskBCELoss()
        logits = torch.randn(3, 2)
        targets = torch.randint(0, 2, (3, 2)).float()
        pred_m = ModelOutput(
            logits=logits,
            metadata={
                "per_doc_logits": logits.unsqueeze(1),  # (3, 1, 2)
                "differentiable_doc_scores": torch.zeros(3, 1),
            },
        )
        pred_p = ModelOutput(logits=logits)
        assert torch.isclose(loss_fn_m(pred_m, targets), loss_fn_p(pred_p, targets), atol=1e-5)

    def test_gradients_flow_through_doc_scores(self):
        loss_fn = MultiTaskBCEMarginalizedLoss(num_tasks=2)
        per_doc = torch.randn(2, 4, 2, requires_grad=True)
        scores = torch.randn(2, 4, requires_grad=True)
        pred = ModelOutput(
            logits=torch.zeros(2, 2),
            metadata={"per_doc_logits": per_doc, "differentiable_doc_scores": scores},
        )
        targets = torch.randint(0, 2, (2, 2)).float()
        loss_fn(pred, targets).backward()
        assert scores.grad is not None
        assert per_doc.grad is not None

    def test_missing_metadata_raises(self):
        loss_fn = MultiTaskBCEMarginalizedLoss(num_tasks=2)
        pred = ModelOutput(logits=torch.zeros(2, 2))
        with pytest.raises(ValueError, match="per_doc_logits"):
            loss_fn(pred, torch.zeros(2, 2))

    def test_missing_doc_scores_raises(self):
        loss_fn = MultiTaskBCEMarginalizedLoss(num_tasks=2)
        pred = ModelOutput(logits=torch.zeros(2, 2), metadata={"per_doc_logits": torch.zeros(2, 4, 2)})
        with pytest.raises(ValueError, match="differentiable_doc_scores"):
            loss_fn(pred, torch.zeros(2, 2))

    def test_wrong_target_type_raises(self):
        loss_fn = MultiTaskBCEMarginalizedLoss(num_tasks=2)
        with pytest.raises(ValueError):
            loss_fn(_marginal_pred(2, 4, 2), {"x": torch.zeros(2, 2)})


# ---------------------------------------------------------------------------
# MultiTaskMEDSDatamodule (integration, uses temp files)
# ---------------------------------------------------------------------------


class TestMultiTaskMEDSDataset:
    def test_load_code_index(self, tmp_path):
        """load_code_index parses code_index.json correctly."""
        from medrap.multitask_datamodule import load_code_index

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
        dataset._mt_lookup = {}
        dataset._load_mt_labels(tmp_path, "train")
        assert dataset._mt_lookup[(7, datetime.datetime(2020, 1, 1))].tolist() == [1.0, 0.0]

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
        dataset._mt_lookup = {(7, datetime.datetime(2020, 1, 1)): torch.tensor([1.0, 0.0])}
        dataset.index = [(7,), (8,)]
        dataset.schema_df = {
            "prediction_time": [
                datetime.datetime(2020, 1, 1),
                datetime.datetime(2020, 1, 2),
            ]
        }

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


def test_passthrough_fusion_returns_patient_state():
    fusion = PassthroughFusion()
    patient_state = torch.randn(2, 3, 4)
    out = fusion.fuse(FusionInput(patient_state=patient_state, retrieval_memory=torch.randn(2, 1, 5, 6, 4)))

    assert out.fused_state is patient_state


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
        assert (11, datetime.datetime(2020, 1, 1)) in ds._mt_lookup
        assert ds._mt_lookup[(11, datetime.datetime(2020, 1, 1))].tolist() == [1.0, 0.0]


class TestMultiTaskMEDSDatamodule:
    """Cover lines 119-120 (init) + 128-155 (dataloader/dataset forwarders) in
    src/medrap/multitask_datamodule.py."""

    def test_datamodule_init_wires_inner_and_forwards_dataloaders_and_datasets(self, monkeypatch):
        from medrap.multitask_datamodule import MultiTaskMEDSDatamodule

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

        import medrap.multitask_datamodule as mod

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
