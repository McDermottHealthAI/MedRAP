"""Supervised task objects for training wrappers."""

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn

type TaskTargets = Tensor | dict[str, Tensor]


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
    def loss(self, logits: Tensor, targets: TaskTargets) -> Tensor:
        """Compute the task loss from model logits and task targets.

        Args:
            logits: Model logits for the current minibatch.
            targets: Task targets returned by :meth:`extract_targets`.

        Returns:
            Tensor: Scalar loss tensor with shape ``()``.
        """

    @abstractmethod
    def metrics(self, logits: Tensor, targets: TaskTargets) -> Mapping[str, Tensor]:
        """Return scalar task metrics derived from logits and targets.

        Args:
            logits: Model logits for the current minibatch.
            targets: Task targets returned by :meth:`extract_targets`.

        Returns:
            Mapping[str, Tensor]: Scalar metric tensors keyed by metric name.
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
        computes BCE-with-logits loss from logits shaped ``(B, 1)``, and reports
        scalar accuracy metrics.
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

    @staticmethod
    def _flatten_logits(logits: Tensor) -> Tensor:
        if logits.ndim == 2 and logits.shape[1] == 1:
            return logits.squeeze(1)
        raise ValueError(f"BinaryClassificationTask expects logits shaped (B, 1); got {tuple(logits.shape)}")

    @staticmethod
    def _flatten_targets(targets: Tensor) -> Tensor:
        if targets.ndim == 1:
            return targets.float()
        raise ValueError(f"BinaryClassificationTask expects targets shaped (B,); got {tuple(targets.shape)}")

    @staticmethod
    def _require_tensor_targets(targets: TaskTargets) -> Tensor:
        if isinstance(targets, Tensor):
            return targets
        raise ValueError("BinaryClassificationTask expects tensor targets, not structured targets.")

    def loss(self, logits: Tensor, targets: TaskTargets) -> Tensor:
        """Compute BCE-with-logits loss.

        Args:
            logits: Model logits with shape ``(B, 1)``.
            targets: Binary targets with shape ``(B,)``.

        Returns:
            Tensor: Scalar loss tensor with shape ``()``.

        Examples:
            >>> import torch
            >>> task = BinaryClassificationTask()
            >>> logits = torch.FloatTensor([[0.0], [2.0]])
            >>> targets = torch.BoolTensor([False, True])
            >>> round(float(task.loss(logits, targets)), 4)
            0.41
        """
        flat_logits = self._flatten_logits(logits)
        flat_targets = self._flatten_targets(self._require_tensor_targets(targets))
        return torch.nn.functional.binary_cross_entropy_with_logits(
            flat_logits,
            flat_targets,
        )

    def metrics(self, logits: Tensor, targets: TaskTargets) -> Mapping[str, Tensor]:
        """Return binary-accuracy metrics derived from logits.

        Args:
            logits: Model logits with shape ``(B, 1)``.
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
        flat_logits = self._flatten_logits(logits)
        flat_targets = self._flatten_targets(self._require_tensor_targets(targets)).bool()
        predictions = flat_logits >= 0
        return {"accuracy": (predictions == flat_targets).float().mean()}
