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
    model_output_binary_model: nn.Module,
) -> None:
    module = MedRAPSupervisedLightningModule(
        model=model_output_binary_model,
        task=BinaryClassificationTask(),
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


def test_lightning_module_supports_structured_task_targets(
    supervised_batch: MEDSTorchBatch,
    model_output_binary_model: nn.Module,
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
            self, predictions: ModelOutput, targets: torch.Tensor | dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            assert isinstance(targets, dict)
            predicted_labels = predictions.logits.squeeze(1) >= 0
            labels = targets["labels"].bool()
            return {"accuracy": (predicted_labels == labels).float().mean()}

    class StructuredBinaryLoss(SupervisedLoss):
        def forward(
            self, predictions: ModelOutput, targets: torch.Tensor | dict[str, torch.Tensor]
        ) -> torch.Tensor:
            assert isinstance(targets, dict)
            return torch.nn.functional.binary_cross_entropy_with_logits(
                predictions.logits.squeeze(1),
                targets["labels"],
            )

    module = MedRAPSupervisedLightningModule(
        model=model_output_binary_model,
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


def test_training_step_uses_batch_size_fallback_without_batch_size() -> None:
    class SimpleBatch:
        def __init__(self) -> None:
            self.code = torch.LongTensor([[1, 2, 3], [3, 2, 1]])
            self.boolean_value = torch.BoolTensor([True, False])

    class SimpleBinaryModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 1)

        def forward(self, batch: object) -> ModelOutput:
            return ModelOutput(logits=self.linear(batch.code.float()))

    module = MedRAPSupervisedLightningModule(model=SimpleBinaryModel(), task=BinaryClassificationTask())

    loss = module._run_supervised_step(SimpleBatch(), stage="train")

    assert loss.ndim == 0


def test_lightning_module_supports_custom_loss_over_model_output_metadata(
    supervised_batch: MEDSTorchBatch,
    model_output_binary_model: nn.Module,
) -> None:
    class MetadataLoss(SupervisedLoss):
        def forward(
            self, predictions: ModelOutput, targets: torch.Tensor | dict[str, torch.Tensor]
        ) -> torch.Tensor:
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


def test_lightning_module_test_step_runs(
    supervised_batch: MEDSTorchBatch,
    model_output_binary_model: nn.Module,
) -> None:
    module = MedRAPSupervisedLightningModule(
        model=model_output_binary_model,
        task=BinaryClassificationTask(),
    )
    trainer = lightning.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_test_batches=1,
    )
    dataloader = DataLoader([supervised_batch], batch_size=None)

    metrics = trainer.test(module, dataloaders=dataloader)

    assert len(metrics) == 1
    assert "test/loss" in metrics[0]


def test_predict_step_returns_targets_when_available(
    supervised_batch: MEDSTorchBatch,
    model_output_binary_model: nn.Module,
) -> None:
    module = MedRAPSupervisedLightningModule(
        model=model_output_binary_model,
        task=BinaryClassificationTask(),
    )

    result = module.predict_step(supervised_batch, batch_idx=0)

    assert "logits" in result
    assert "targets" in result
    assert result["logits"].device.type == "cpu"
    assert result["targets"].device.type == "cpu"


def test_predict_step_skips_targets_when_extract_raises(
    model_output_binary_model: nn.Module,
) -> None:
    """Covers the ``except Exception: pass`` branch in ``predict_step``."""
    # A batch without boolean_value makes BinaryClassificationTask.extract_targets raise.
    batch_without_labels = MEDSTorchBatch(
        code=torch.LongTensor([[1, 2, 3], [3, 2, 1]]),
        numeric_value=torch.zeros((2, 3), dtype=torch.float32),
        numeric_value_mask=torch.zeros((2, 3), dtype=torch.bool),
        time_delta_days=torch.zeros((2, 3), dtype=torch.float32),
    )
    module = MedRAPSupervisedLightningModule(
        model=model_output_binary_model,
        task=BinaryClassificationTask(),
    )

    result = module.predict_step(batch_without_labels, batch_idx=0)

    assert "logits" in result
    assert "targets" not in result


def test_configure_optimizers_with_warmup_returns_scheduler_dict(
    supervised_batch: MEDSTorchBatch,
    model_output_binary_model: nn.Module,
) -> None:
    module = MedRAPSupervisedLightningModule(
        model=model_output_binary_model,
        task=BinaryClassificationTask(),
        warmup_steps=10,
    )
    trainer = lightning.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_train_batches=2,
    )
    dataloader = DataLoader([supervised_batch, supervised_batch], batch_size=None)
    trainer.fit(module, train_dataloaders=dataloader)

    result = module.configure_optimizers()
    assert isinstance(result, dict)
    assert "optimizer" in result
    assert "lr_scheduler" in result
    assert result["lr_scheduler"]["interval"] == "step"


def test_configure_optimizers_without_warmup_returns_optimizer(
    supervised_batch: MEDSTorchBatch,
    model_output_binary_model: nn.Module,
) -> None:
    module = MedRAPSupervisedLightningModule(
        model=model_output_binary_model,
        task=BinaryClassificationTask(),
        warmup_steps=0,
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

    result = module.configure_optimizers()
    assert not isinstance(result, dict)


def test_predict_step_captures_marginalized_tensors_from_metadata(
    supervised_batch: MEDSTorchBatch,
) -> None:
    """Covers the ``per_doc_logits`` / ``differentiable_doc_scores`` branch."""

    class MarginalizedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer_norm = nn.LayerNorm(3)
            self.linear = nn.Linear(3, 1)

        def forward(self, batch: MEDSTorchBatch) -> ModelOutput:
            features = self.layer_norm(batch.code.float())
            logits = self.linear(features)
            # Shape chosen arbitrarily; predict_step only checks isinstance(value, Tensor).
            return ModelOutput(
                logits=logits,
                metadata={
                    "per_doc_logits": logits.unsqueeze(-1),
                    "differentiable_doc_scores": logits.squeeze(-1),
                    "per_doc_logits_non_tensor": "skip-me",
                },
            )

    module = MedRAPSupervisedLightningModule(
        model=MarginalizedModel(),
        task=BinaryClassificationTask(),
    )

    result = module.predict_step(supervised_batch, batch_idx=0)

    assert "per_doc_logits" in result
    assert "differentiable_doc_scores" in result
    assert result["per_doc_logits"].device.type == "cpu"
    assert result["differentiable_doc_scores"].device.type == "cpu"
