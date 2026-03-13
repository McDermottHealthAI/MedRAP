"""Supervised PyTorch Lightning wrapper for MedRAP."""

from collections.abc import Callable

import lightning
import torch
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn
from torch.optim import Optimizer

from .task import (
    BinaryClassificationLoss,
    BinaryClassificationTask,
    SupervisedLoss,
    SupervisedTask,
    TaskPredictions,
)
from .types import ModelOutput


class MedRAPSupervisedLightningModule(lightning.LightningModule):
    """Supervised Lightning wrapper around a plain RAP model.

    Args:
        model: Plain PyTorch model returning ``ModelOutput`` or logits.
        task: Supervised task object.
        loss_fn: Supervised loss object.
        optimizer: Optimizer factory taking grouped parameter configs.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        task: SupervisedTask | None = None,
        loss_fn: SupervisedLoss | None = None,
        optimizer: Callable[[list[dict[str, object]]], Optimizer] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.task = task or BinaryClassificationTask()
        self.loss_fn = loss_fn or BinaryClassificationLoss()
        self.optimizer_factory = optimizer or (
            lambda params: torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
        )

    def forward(self, batch: MEDSTorchBatch) -> Tensor | ModelOutput:
        """Run the wrapped plain model on a MEDS batch.

        Args:
            batch: Input ``MEDSTorchBatch``.

        Returns:
            Tensor | ModelOutput: Wrapped model output for a batch of size ``B``.
            When the plain model returns a tensor, it is expected to have shape
            ``(B, D)`` where ``D`` is the task output width. When the plain model
            returns ``ModelOutput``, its ``logits`` field is expected to have the
            same shape ``(B, D)``.

        Examples:
            >>> module = MedRAPSupervisedLightningModule(model=ModelOutputBinaryModel())
            >>> output = module.forward(make_supervised_batch())
            >>> isinstance(output, ModelOutput)
            True
            >>> tuple(output.logits.shape)
            (2, 1)
        """
        return self.model(batch)

    def _iter_no_decay_names(self) -> set[str]:
        no_decay_names: set[str] = set()
        norm_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LayerNorm,
            nn.LocalResponseNorm,
        )
        for module_name, module in self.named_modules():
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                if param_name == "bias" or (isinstance(module, norm_modules) and param_name == "weight"):
                    no_decay_names.add(full_name)
        return no_decay_names

    def _grouped_parameters(self) -> list[dict[str, object]]:
        no_decay_names = self._iter_no_decay_names()
        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []

        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name in no_decay_names:
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

        groups: list[dict[str, object]] = []
        if decay_params:
            groups.append({"params": decay_params})
        if no_decay_params:
            groups.append({"params": no_decay_params, "weight_decay": 0.0})
        return groups

    def _run_supervised_step(self, raw_batch: MEDSTorchBatch, *, stage: str) -> Tensor:
        predictions: TaskPredictions = self.forward(raw_batch)
        targets = self.task.extract_targets(raw_batch)
        loss = self.loss_fn(predictions, targets)

        batch_size = getattr(raw_batch, "batch_size", None)
        if not isinstance(batch_size, int):
            batch_size = (
                predictions.shape[0] if isinstance(predictions, Tensor) else predictions.logits.shape[0]
            )

        is_train = stage == "train"
        self.log(
            f"{stage}/loss",
            loss,
            on_step=is_train,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        for metric_name, metric_value in self.task.metrics(predictions, targets).items():
            self.log(
                f"{stage}/{metric_name}",
                metric_value,
                on_step=is_train,
                on_epoch=True,
                prog_bar=not is_train,
                batch_size=batch_size,
            )
        return loss

    def training_step(self, batch: MEDSTorchBatch, _batch_idx: int) -> Tensor:
        """Compute the supervised training loss for one batch.

        Args:
            batch: Input ``MEDSTorchBatch`` with ``boolean_value`` targets.
            _batch_idx: Unused batch index required by Lightning.

        Returns:
            Tensor: Scalar training loss with shape ``()``.
        """
        return self._run_supervised_step(batch, stage="train")

    def validation_step(self, batch: MEDSTorchBatch, _batch_idx: int) -> Tensor:
        """Compute the supervised validation loss for one batch.

        Args:
            batch: Input ``MEDSTorchBatch`` with ``boolean_value`` targets.
            _batch_idx: Unused batch index required by Lightning.

        Returns:
            Tensor: Scalar validation loss with shape ``()``.
        """
        return self._run_supervised_step(batch, stage="val")

    def test_step(self, batch: MEDSTorchBatch, _batch_idx: int) -> Tensor:
        """Compute the supervised test loss for one batch.

        Args:
            batch: Input ``MEDSTorchBatch`` with ``boolean_value`` targets.
            _batch_idx: Unused batch index required by Lightning.

        Returns:
            Tensor: Scalar test loss with shape ``()``.
        """
        return self._run_supervised_step(batch, stage="test")

    def configure_optimizers(self) -> Optimizer:
        """Construct the optimizer for the wrapped plain model.

        Returns:
            Configured optimizer with grouped weight decay.

        Examples:
            >>> module = MedRAPSupervisedLightningModule(model=TensorBinaryModel())
            >>> optimizer = module.configure_optimizers()
            >>> isinstance(optimizer, torch.optim.AdamW)
            True
            >>> len(optimizer.param_groups)
            2
        """
        return self.optimizer_factory(self._grouped_parameters())
