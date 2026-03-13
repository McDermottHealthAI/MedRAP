import lightning
import pytest
import torch
from meds_torchdata import MEDSTorchBatch
from torch import nn
from torch.utils.data import DataLoader

from medrap.lightning_module import MedRAPSupervisedLightningModule
from medrap.task import BinaryClassificationTask, SupervisedLoss, SupervisedTask
from medrap.types import ModelOutput


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


@pytest.fixture
def model_output_binary_model() -> nn.Module:
    class ModelOutputBinaryModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer_norm = nn.LayerNorm(3)
            self.linear = nn.Linear(3, 1)

        def forward(self, batch: MEDSTorchBatch) -> ModelOutput:
            features = self.layer_norm(batch.code.float())
            logits = self.linear(features)
            return ModelOutput(logits=logits, metadata={"extra": logits.square()})

    return ModelOutputBinaryModel()


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

        def metrics(
            self, predictions: torch.Tensor | ModelOutput, targets: torch.Tensor | dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            assert isinstance(targets, dict)
            assert isinstance(predictions, torch.Tensor)
            predicted_labels = predictions.squeeze(1) >= 0
            labels = targets["labels"].bool()
            return {"accuracy": (predicted_labels == labels).float().mean()}

    class StructuredBinaryLoss(SupervisedLoss):
        def forward(
            self, predictions: torch.Tensor | ModelOutput, targets: torch.Tensor | dict[str, torch.Tensor]
        ) -> torch.Tensor:
            assert isinstance(targets, dict)
            assert isinstance(predictions, torch.Tensor)
            return torch.nn.functional.binary_cross_entropy_with_logits(
                predictions.squeeze(1),
                targets["labels"],
            )

    module = MedRAPSupervisedLightningModule(
        model=tensor_binary_model,
        task=StructuredBinaryTask(),
        loss_fn=StructuredBinaryLoss(),
    )
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

        def metrics(
            self, predictions: torch.Tensor | ModelOutput, targets: torch.Tensor | dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return {}

    class LearnableLoss(SupervisedLoss):
        def __init__(self, task: LearnableTask) -> None:
            super().__init__()
            self.task = task

        def forward(
            self, predictions: torch.Tensor | ModelOutput, targets: torch.Tensor | dict[str, torch.Tensor]
        ) -> torch.Tensor:
            assert isinstance(predictions, torch.Tensor)
            assert isinstance(targets, torch.Tensor)
            return torch.nn.functional.binary_cross_entropy_with_logits(
                self.task.scale * predictions.squeeze(1),
                targets,
            )

    task = LearnableTask()
    module = MedRAPSupervisedLightningModule(
        model=tensor_binary_model,
        task=task,
        loss_fn=LearnableLoss(task),
    )
    optimizer = module.configure_optimizers()
    optimized_params = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}

    assert id(module.task.scale) in optimized_params


def test_lightning_module_supports_custom_loss_over_model_output_metadata(
    supervised_batch: MEDSTorchBatch,
    model_output_binary_model: nn.Module,
) -> None:
    class MetadataLoss(SupervisedLoss):
        def forward(
            self, predictions: torch.Tensor | ModelOutput, targets: torch.Tensor | dict[str, torch.Tensor]
        ) -> torch.Tensor:
            assert isinstance(predictions, ModelOutput)
            assert isinstance(targets, torch.Tensor)
            extra = predictions.metadata["extra"]
            assert isinstance(extra, torch.Tensor)
            return predictions.logits.square().mean() + extra.mean() + targets.mean()

    module = MedRAPSupervisedLightningModule(
        model=model_output_binary_model,
        task=BinaryClassificationTask(),
        loss_fn=MetadataLoss(),
    )
    trainer = lightning.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_train_batches=1,
    )
    dataloader = DataLoader([supervised_batch], batch_size=None)

    trainer.fit(module, train_dataloaders=dataloader)

    assert trainer.callback_metrics["train/loss"].ndim == 0
