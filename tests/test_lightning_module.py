import lightning
import pytest
import torch
from meds_torchdata import MEDSTorchBatch
from torch import nn
from torch.utils.data import DataLoader

from medrap.lightning_module import MedRAPSupervisedLightningModule
from medrap.task import BinaryClassificationTask, SupervisedTask


@pytest.fixture
def supervised_batch() -> MEDSTorchBatch:
    batch = MEDSTorchBatch(
        code=torch.LongTensor([[1, 2, 3], [3, 2, 1]]),
        numeric_value=torch.zeros((2, 3), dtype=torch.float32),
        numeric_value_mask=torch.zeros((2, 3), dtype=torch.bool),
        time_delta_days=torch.zeros((2, 3), dtype=torch.float32),
    )
    batch.boolean_value = torch.BoolTensor([True, False])
    return batch


@pytest.fixture
def tensor_binary_model() -> nn.Module:
    class TensorBinaryModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer_norm = nn.LayerNorm(3)
            self.linear = nn.Linear(3, 1)

        def forward(self, batch: MEDSTorchBatch) -> torch.Tensor:
            features = self.layer_norm(batch.code.float())
            return self.linear(features)

    return TensorBinaryModel()


def test_lightning_module_trainer_smoke(
    supervised_batch: MEDSTorchBatch,
    tensor_binary_model: nn.Module,
) -> None:
    module = MedRAPSupervisedLightningModule(model=tensor_binary_model, task=BinaryClassificationTask())
    trainer = lightning.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_train_batches=1,
        limit_val_batches=1,
    )
    dataloader = DataLoader([supervised_batch], batch_size=None)

    trainer.fit(module, train_dataloaders=dataloader, val_dataloaders=dataloader)

    assert trainer.callback_metrics["train/loss"].ndim == 0
    assert trainer.callback_metrics["val/loss"].ndim == 0


def test_lightning_module_supports_structured_task_targets(
    supervised_batch: MEDSTorchBatch,
    tensor_binary_model: nn.Module,
) -> None:
    class StructuredBinaryTask(SupervisedTask):
        def __init__(self) -> None:
            super().__init__(output_dim=1)

        def extract_targets(self, batch: MEDSTorchBatch) -> dict[str, torch.Tensor]:
            return {
                "labels": batch.boolean_value.float(),
                "mask": torch.ones_like(batch.boolean_value, dtype=torch.bool),
            }

        def loss(self, logits: torch.Tensor, targets: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
            assert isinstance(targets, dict)
            return torch.nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(1),
                targets["labels"],
            )

        def metrics(
            self, logits: torch.Tensor, targets: torch.Tensor | dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            assert isinstance(targets, dict)
            predictions = logits.squeeze(1) >= 0
            labels = targets["labels"].bool()
            return {"accuracy": (predictions == labels).float().mean()}

    module = MedRAPSupervisedLightningModule(model=tensor_binary_model, task=StructuredBinaryTask())
    trainer = lightning.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_train_batches=1,
        limit_val_batches=1,
    )
    dataloader = DataLoader([supervised_batch], batch_size=None)

    trainer.fit(module, train_dataloaders=dataloader, val_dataloaders=dataloader)

    assert trainer.callback_metrics["train/loss"].ndim == 0
    assert trainer.callback_metrics["val/loss"].ndim == 0


def test_configure_optimizers_includes_task_parameters(
    supervised_batch: MEDSTorchBatch,
    tensor_binary_model: nn.Module,
) -> None:
    class LearnableTask(SupervisedTask):
        def __init__(self) -> None:
            super().__init__(output_dim=1)
            self.scale = nn.Parameter(torch.ones(()))

        def extract_targets(self, batch: MEDSTorchBatch) -> torch.Tensor:
            return batch.boolean_value.float()

        def loss(self, logits: torch.Tensor, targets: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
            assert isinstance(targets, torch.Tensor)
            return torch.nn.functional.binary_cross_entropy_with_logits(
                self.scale * logits.squeeze(1),
                targets,
            )

        def metrics(
            self, logits: torch.Tensor, targets: torch.Tensor | dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return {}

    module = MedRAPSupervisedLightningModule(model=tensor_binary_model, task=LearnableTask())
    optimizer = module.configure_optimizers()
    optimized_params = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}

    assert id(module.task.scale) in optimized_params
