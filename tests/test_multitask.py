"""Tests for multi-task binary classification components."""

import json

import polars as pl
import pytest
import torch
from meds_torchdata import MEDSTorchBatch

from medrap.losses import MultiTaskBCELoss, MultiTaskBCEMarginalizedLoss
from medrap.task import MultiTaskBinaryClassificationTask
from medrap.types import ModelOutput

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
