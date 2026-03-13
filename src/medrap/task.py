"""Supervised task objects for training wrappers."""

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn

from .types import ModelOutput

type TaskPredictions = Tensor | ModelOutput
type TaskTargets = Tensor | dict[str, Tensor]


def _extract_logits(predictions: TaskPredictions) -> Tensor:
    if isinstance(predictions, Tensor):
        return predictions
    return predictions.logits


def _require_tensor_targets(targets: TaskTargets, *, owner: str) -> Tensor:
    if isinstance(targets, Tensor):
        return targets
    raise ValueError(f"{owner} expects tensor targets, not structured targets.")


def _flatten_binary_logits(predictions: TaskPredictions, *, owner: str) -> Tensor:
    logits = _extract_logits(predictions)
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits.squeeze(1)
    raise ValueError(f"{owner} expects logits shaped (B, 1); got {tuple(logits.shape)}")


def _flatten_binary_targets(targets: TaskTargets, *, owner: str) -> Tensor:
    tensor_targets = _require_tensor_targets(targets, owner=owner)
    if tensor_targets.ndim == 1:
        return tensor_targets.float()
    raise ValueError(f"{owner} expects targets shaped (B,); got {tuple(tensor_targets.shape)}")


class SupervisedTask(nn.Module, ABC):
    """Abstract base for supervised task helpers.

    Args:
        output_dim: Expected final model-output width for this task.
    """

    def __init__(self, *, output_dim: int) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    @abstractmethod
    def extract_targets(self, batch: MEDSTorchBatch) -> TaskTargets:
        """Extract and normalize task targets from a MEDS batch.

        Args:
            batch: Input ``MEDSTorchBatch`` for the current minibatch.

        Returns:
            TaskTargets: Either a tensor target or a structured mapping of tensors.
        """

    @abstractmethod
    def metrics(self, predictions: TaskPredictions, targets: TaskTargets) -> Mapping[str, Tensor]:
        """Return scalar task metrics derived from logits and targets.

        Args:
            predictions: Model predictions for the current minibatch.
            targets: Task targets returned by :meth:`extract_targets`.

        Returns:
            Mapping[str, Tensor]: Scalar metric tensors keyed by metric name.
        """


class SupervisedLoss(nn.Module, ABC):
    """Abstract base for supervised training objectives."""

    @abstractmethod
    def forward(self, predictions: TaskPredictions, targets: TaskTargets) -> Tensor:
        """Compute a scalar training loss from predictions and task targets.

        Args:
            predictions: Model predictions for the current minibatch.
            targets: Task targets returned by ``SupervisedTask.extract_targets``.

        Returns:
            Tensor: Scalar loss tensor with shape ``()``.
        """


class BinaryClassificationTask(SupervisedTask):
    """Binary classification task for scalar logits and boolean labels.

    Args:
        label_field: Batch attribute name containing the binary labels. The field must
            hold a tensor with shape ``(B,)`` or ``(B, 1)``.
        output_dim: Expected model output size. Must be ``1`` so model logits have
            shape ``(B, 1)``.

    Returns:
        BinaryClassificationTask: Task helper that extracts labels from a MEDS batch,
        reports scalar accuracy metrics from predictions with logits shaped
        ``(B, 1)``.
    """

    def __init__(self, *, label_field: str = "boolean_value", output_dim: int = 1) -> None:
        super().__init__(output_dim=output_dim)
        if int(output_dim) != 1:
            raise ValueError(f"BinaryClassificationTask requires output_dim=1, got {output_dim}")
        self.label_field = label_field

    def extract_targets(self, batch: MEDSTorchBatch) -> Tensor:
        """Extract binary targets from the configured batch field.

        Args:
            batch: ``MEDSTorchBatch`` containing ``batch.<label_field>`` with shape
                ``(B,)`` or ``(B, 1)``.

        Returns:
            Tensor: Float tensor of shape ``(B,)`` suitable for BCE-with-logits.

        Examples:
            >>> task = BinaryClassificationTask()
            >>> batch = make_supervised_batch()
            >>> targets = task.extract_targets(batch)
            >>> tuple(targets.shape)
            (2,)
            >>> targets.dtype
            torch.float32
        """
        targets = getattr(batch, self.label_field, None)
        if not isinstance(targets, Tensor):
            raise ValueError(f"Expected {self.label_field} targets on the MEDS batch.")
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        elif targets.ndim != 1:
            raise ValueError(
                f"BinaryClassificationTask expects {self.label_field} shaped (B,) or (B, 1); "
                f"got {tuple(targets.shape)}"
            )
        return targets.float()

    def metrics(self, predictions: TaskPredictions, targets: TaskTargets) -> Mapping[str, Tensor]:
        """Return binary-accuracy metrics derived from logits.

        Args:
            predictions: Model predictions whose logits have shape ``(B, 1)``.
            targets: Binary targets with shape ``(B,)``.

        Returns:
            Mapping[str, Tensor]: Metric dictionary containing ``"accuracy"`` mapped
            to a scalar tensor with shape ``()``.

        Examples:
            >>> import torch
            >>> task = BinaryClassificationTask()
            >>> metrics = task.metrics(torch.FloatTensor([[2.0], [-2.0]]), torch.BoolTensor([True, False]))
            >>> sorted(metrics)
            ['accuracy']
            >>> float(metrics["accuracy"])
            1.0
        """
        flat_logits = _flatten_binary_logits(predictions, owner="BinaryClassificationTask")
        flat_targets = _flatten_binary_targets(targets, owner="BinaryClassificationTask").bool()
        predictions = flat_logits >= 0
        return {"accuracy": (predictions == flat_targets).float().mean()}


class BinaryClassificationLoss(SupervisedLoss):
    """Binary BCE-with-logits loss for scalar binary predictions.

    Returns:
        BinaryClassificationLoss: Loss helper that accepts ``Tensor`` or
        ``ModelOutput`` predictions with logits shaped ``(B, 1)`` and binary
        tensor targets shaped ``(B,)``.
    """

    def forward(self, predictions: TaskPredictions, targets: TaskTargets) -> Tensor:
        """Compute BCE-with-logits loss from binary predictions and targets.

        Args:
            predictions: ``Tensor`` or ``ModelOutput`` predictions with logits
                shaped ``(B, 1)``.
            targets: Binary tensor targets shaped ``(B,)``.

        Returns:
            Tensor: Scalar loss tensor with shape ``()``.

        Examples:
            >>> import torch
            >>> loss_fn = BinaryClassificationLoss()
            >>> predictions = ModelOutput(logits=torch.FloatTensor([[0.0], [2.0]]))
            >>> targets = torch.BoolTensor([False, True])
            >>> round(float(loss_fn(predictions, targets)), 4)
            0.41
        """
        flat_logits = _flatten_binary_logits(predictions, owner="BinaryClassificationLoss")
        flat_targets = _flatten_binary_targets(targets, owner="BinaryClassificationLoss")
        return torch.nn.functional.binary_cross_entropy_with_logits(flat_logits, flat_targets)
