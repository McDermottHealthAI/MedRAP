import lightning
import pytest
import torch
from meds_torchdata import MEDSTorchBatch
from torch import nn
from torch.utils.data import DataLoader

from medrap.lightning_module import MedRAPSupervisedLightningModule
from medrap.task import BinaryClassificationTask, SupervisedLoss, SupervisedTask
from medrap.types import ModelOutput, QueryOutput, RetrieverOutput


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


def test_lightning_module_logs_curated_diagnostics(
    supervised_batch: MEDSTorchBatch,
) -> None:
    class DiagnosticModel(nn.Module):
        def forward(self, batch: MEDSTorchBatch) -> ModelOutput:
            return ModelOutput(
                logits=torch.FloatTensor([[0.0], [1.0]]),
                metadata={
                    "query_output": QueryOutput(torch.FloatTensor([[[1.0, 0.0]], [[0.0, 1.0]]])),
                    "retriever_output": RetrieverOutput(
                        doc_tokens=torch.ones(2, 1, 2, 3, dtype=torch.long),
                        doc_attention_mask=torch.ones(2, 1, 2, 3, dtype=torch.bool),
                        doc_ids=torch.LongTensor([[[1, 2]], [[2, 3]]]),
                        doc_scores=torch.FloatTensor([[[2.0, 1.0]], [[0.5, 0.25]]]),
                    ),
                },
            )

    module = MedRAPSupervisedLightningModule(
        model=DiagnosticModel(),
        task=BinaryClassificationTask(),
        diagnostics_every_n_steps=1,
    )
    logged: list[str] = []
    module.log_dict = lambda metrics, *a, **k: logged.extend(metrics)

    loss = module._run_supervised_step(supervised_batch, stage="train")

    assert loss.ndim == 0
    assert "prediction/train/logits_mean" in logged
    assert "query/train/norm_mean" in logged
    assert "retrieval/train/unique_doc_ratio" in logged
    assert "mask/train/pad_fraction" in logged
    assert not any(name.startswith("train/") for name in logged)


def test_lightning_module_can_disable_diagnostics(
    supervised_batch: MEDSTorchBatch,
    model_output_binary_model: nn.Module,
) -> None:
    module = MedRAPSupervisedLightningModule(
        model=model_output_binary_model,
        task=BinaryClassificationTask(),
        diagnostics_every_n_steps=0,
    )
    logged: list[str] = []
    module.log_dict = lambda metrics, *a, **k: logged.extend(metrics)

    loss = module._run_supervised_step(supervised_batch, stage="train")

    assert loss.ndim == 0
    assert logged == []


def test_validation_loop_logs_binary_auroc() -> None:
    class CodeLogitModel(nn.Module):
        def forward(self, batch: MEDSTorchBatch) -> ModelOutput:
            return ModelOutput(logits=batch.code[:, :1].float())

    low_risk = MEDSTorchBatch(
        code=torch.LongTensor([[-2, 0, 0]]),
        numeric_value=torch.zeros((1, 3), dtype=torch.float32),
        numeric_value_mask=torch.zeros((1, 3), dtype=torch.bool),
        time_delta_days=torch.zeros((1, 3), dtype=torch.float32),
    )
    low_risk.boolean_value = torch.BoolTensor([False])
    high_risk = MEDSTorchBatch(
        code=torch.LongTensor([[2, 0, 0]]),
        numeric_value=torch.zeros((1, 3), dtype=torch.float32),
        numeric_value_mask=torch.zeros((1, 3), dtype=torch.bool),
        time_delta_days=torch.zeros((1, 3), dtype=torch.float32),
    )
    high_risk.boolean_value = torch.BoolTensor([True])
    module = MedRAPSupervisedLightningModule(
        model=CodeLogitModel(),
        task=BinaryClassificationTask(),
        validation_auroc=True,
    )
    trainer = lightning.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.validate(module, dataloaders=DataLoader([low_risk, high_risk], batch_size=None))

    assert trainer.callback_metrics["val/auroc"].item() == pytest.approx(1.0)
    assert module._validation_auroc_logits == []
    assert module._validation_auroc_targets == []


def test_validation_loop_logs_multitask_mean_auroc() -> None:
    class MultitaskTask(SupervisedTask):
        def __init__(self) -> None:
            super().__init__(output_dim=2)

        def extract_targets(self, batch: MEDSTorchBatch) -> torch.Tensor:
            return batch.boolean_value.float()

        def metrics(self, predictions: ModelOutput, targets: torch.Tensor | dict[str, torch.Tensor]) -> dict:
            return {}

    class MultitaskLoss(SupervisedLoss):
        def forward(
            self, predictions: ModelOutput, targets: torch.Tensor | dict[str, torch.Tensor]
        ) -> torch.Tensor:
            assert isinstance(targets, torch.Tensor)
            return torch.nn.functional.binary_cross_entropy_with_logits(predictions.logits, targets.float())

    class CodeMultitaskModel(nn.Module):
        def forward(self, batch: MEDSTorchBatch) -> ModelOutput:
            return ModelOutput(logits=batch.code[:, :2].float())

    def _batch(logits: list[int], labels: list[bool]) -> MEDSTorchBatch:
        batch = MEDSTorchBatch(
            code=torch.LongTensor([[*logits, 0]]),
            numeric_value=torch.zeros((1, 3), dtype=torch.float32),
            numeric_value_mask=torch.zeros((1, 3), dtype=torch.bool),
            time_delta_days=torch.zeros((1, 3), dtype=torch.float32),
        )
        batch.boolean_value = torch.BoolTensor([labels])
        return batch

    module = MedRAPSupervisedLightningModule(
        model=CodeMultitaskModel(),
        task=MultitaskTask(),
        loss_fn=MultitaskLoss(),
        validation_auroc=True,
        validation_auroc_log_per_task=True,
    )
    trainer = lightning.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    dataloader = DataLoader(
        [
            _batch([-2, 2], [False, True]),
            _batch([2, -2], [True, False]),
            _batch([-1, -1], [False, False]),
            _batch([1, 1], [True, True]),
        ],
        batch_size=None,
    )

    trainer.validate(module, dataloaders=dataloader)

    assert trainer.callback_metrics["val/auroc/mean"].item() == pytest.approx(1.0)
    assert trainer.callback_metrics["val/auroc/task_0"].item() == pytest.approx(1.0)
    assert trainer.callback_metrics["val/auroc/task_1"].item() == pytest.approx(1.0)


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


def test_grouped_parameters_excludes_vector_parameters_from_weight_decay() -> None:
    class AttentionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim=4, num_heads=2, batch_first=True)
            self.linear = nn.Linear(4, 1)

        def forward(self, batch: MEDSTorchBatch) -> ModelOutput:
            features = batch.code.float().unsqueeze(-1).expand(-1, -1, 4)
            attended, _ = self.attn(features, features, features, need_weights=False)
            return ModelOutput(logits=self.linear(attended.mean(dim=1)))

    model = AttentionModel()
    module = MedRAPSupervisedLightningModule(model=model)

    decay_group, no_decay_group = module._grouped_parameters()
    decay_params = set(decay_group["params"])
    no_decay_params = set(no_decay_group["params"])

    assert model.attn.in_proj_weight in decay_params
    assert model.attn.in_proj_bias in no_decay_params
    assert model.linear.weight in decay_params
    assert model.linear.bias in no_decay_params


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


# ---------------------------------------------------------------------------
# Coverage gap-filler for callbacks.EndOfFitValAUROCCallback (line 452)
# ---------------------------------------------------------------------------


def test_end_of_fit_val_auroc_callback_skips_when_single_class_targets() -> None:
    """All-same-class targets should trigger the early-return at line 452 of
    ``src/medrap/callbacks.py`` (``unique.numel() < 2``)."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from medrap.callbacks import EndOfFitValAUROCCallback
    from medrap.types import ModelOutput

    def _auroc_batch(y: bool) -> MEDSTorchBatch:
        b = MEDSTorchBatch(
            code=torch.LongTensor([[1]]),
            numeric_value=torch.zeros((1, 1), dtype=torch.float32),
            numeric_value_mask=torch.zeros((1, 1), dtype=torch.bool),
            time_delta_days=torch.zeros((1, 1), dtype=torch.float32),
        )
        b.boolean_value = torch.BoolTensor([y])
        return b

    cb = EndOfFitValAUROCCallback()
    logger = SimpleNamespace(log_metrics=MagicMock())
    plm = MagicMock(spec=lightning.LightningModule)
    plm.training = True
    plm.device = torch.device("cpu")
    plm.eval = MagicMock()
    plm.train = MagicMock()
    plm.transfer_batch_to_device = lambda batch, device, dataloader_idx=0: batch
    plm.task = BinaryClassificationTask()
    plm.side_effect = [
        ModelOutput(logits=torch.tensor([[0.0]])),
        ModelOutput(logits=torch.tensor([[1.0]])),
    ]
    cb.on_fit_end(
        SimpleNamespace(
            sanity_checking=False,
            global_step=0,
            loggers=[logger],
            datamodule=None,
            # Both batches have the same label ``False`` → unique.numel() < 2 → return.
            val_dataloaders=DataLoader(
                [_auroc_batch(False), _auroc_batch(False)], batch_size=None
            ),
        ),
        plm,
    )
    # No metric was logged because we short-circuited before computing AUROC.
    logger.log_metrics.assert_not_called()
