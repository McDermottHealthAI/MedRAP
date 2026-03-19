import pytest
import torch
from meds_torchdata import MEDSTorchBatch

from medrap.datamodule import SyntheticSupervisedDatamodule


def _batch() -> MEDSTorchBatch:
    return MEDSTorchBatch(
        code=torch.LongTensor([[1, 2, 3], [3, 2, 1]]),
        numeric_value=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numeric_value_mask=torch.BoolTensor([[False, False, False], [False, False, False]]),
        time_delta_days=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        boolean_value=torch.BoolTensor([True, False]),
    )


def test_synthetic_supervised_datamodule_defaults_val_and_test_to_train_batch() -> None:
    datamodule = SyntheticSupervisedDatamodule(
        train_batch=_batch(),
        train_repeat=2,
        val_repeat=1,
        test_repeat=1,
    )

    train_batch = next(iter(datamodule.train_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))
    test_batch = next(iter(datamodule.test_dataloader()))

    assert tuple(train_batch.code.shape) == (2, 3)
    assert tuple(val_batch.code.shape) == (2, 3)
    assert tuple(test_batch.code.shape) == (2, 3)
    assert tuple(train_batch.boolean_value.shape) == (2,)


def test_synthetic_supervised_datamodule_rejects_non_positive_repeat() -> None:
    datamodule = SyntheticSupervisedDatamodule(train_batch=_batch(), train_repeat=0)

    with pytest.raises(ValueError, match="repeat must be at least 1"):
        datamodule.train_dataloader()
