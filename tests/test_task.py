import pytest
import torch
from meds_torchdata import MEDSTorchBatch

from medrap.task import BinaryClassificationLoss, BinaryClassificationTask
from medrap.types import ModelOutput


def _example_batch() -> MEDSTorchBatch:
    batch = MEDSTorchBatch(
        code=torch.LongTensor([[1, 2, 3], [3, 2, 1]]),
        numeric_value=torch.zeros((2, 3), dtype=torch.float32),
        numeric_value_mask=torch.zeros((2, 3), dtype=torch.bool),
        time_delta_days=torch.zeros((2, 3), dtype=torch.float32),
    )
    batch.boolean_value = torch.BoolTensor([True, False])
    return batch


def test_binary_classification_task_rejects_non_scalar_output_dim() -> None:
    with pytest.raises(ValueError, match="requires output_dim=1"):
        BinaryClassificationTask(output_dim=2)


def test_binary_classification_task_requires_label_field_on_batch() -> None:
    batch = MEDSTorchBatch(
        code=torch.LongTensor([[1, 2, 3], [3, 2, 1]]),
        numeric_value=torch.zeros((2, 3), dtype=torch.float32),
        numeric_value_mask=torch.zeros((2, 3), dtype=torch.bool),
        time_delta_days=torch.zeros((2, 3), dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="Expected boolean_value targets"):
        BinaryClassificationTask().extract_targets(batch)


def test_binary_classification_task_rejects_invalid_target_rank() -> None:
    batch = _example_batch()
    batch.boolean_value = torch.ones((2, 2), dtype=torch.bool)

    with pytest.raises(ValueError, match="expects boolean_value shaped"):
        BinaryClassificationTask().extract_targets(batch)


def test_binary_classification_task_squeezes_singleton_target_axis() -> None:
    batch = _example_batch()
    batch.boolean_value = torch.BoolTensor([[True], [False]])

    targets = BinaryClassificationTask().extract_targets(batch)

    assert tuple(targets.shape) == (2,)


def test_binary_classification_task_metrics_rejects_invalid_logit_shape() -> None:
    with pytest.raises(ValueError, match="expects logits shaped \\(B, 1\\)"):
        BinaryClassificationTask().metrics(torch.FloatTensor([1.0, -1.0]), torch.BoolTensor([True, False]))


def test_binary_classification_loss_rejects_structured_targets() -> None:
    loss_fn = BinaryClassificationLoss()
    predictions = ModelOutput(logits=torch.FloatTensor([[0.0], [1.0]]))

    with pytest.raises(ValueError, match="expects tensor targets"):
        loss_fn(predictions, {"labels": torch.BoolTensor([True, False])})


def test_binary_classification_loss_rejects_invalid_target_shape() -> None:
    loss_fn = BinaryClassificationLoss()
    predictions = ModelOutput(logits=torch.FloatTensor([[0.0], [1.0]]))

    with pytest.raises(ValueError, match="expects targets shaped \\(B,\\)"):
        loss_fn(predictions, torch.BoolTensor([[True], [False]]))
