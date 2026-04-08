"""Supervised task objects for training wrappers."""

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn

from .types import ModelOutput

type TaskTargets = Tensor | dict[str, Tensor]


def _flatten_binary_logits(predictions: ModelOutput, *, owner: str) -> Tensor:
    logits = predictions.logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits.squeeze(1)
    raise ValueError(f"{owner} expects logits shaped (B, 1); got {tuple(logits.shape)}")


def _flatten_binary_targets(targets: TaskTargets, *, owner: str) -> Tensor:
    if not isinstance(targets, Tensor):
        raise ValueError(f"{owner} expects tensor targets, not structured targets.")
    tensor_targets = targets
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
        self.output_dim = output_dim

    @abstractmethod
    def extract_targets(self, batch: MEDSTorchBatch) -> TaskTargets:
        """Extract and normalize task targets from a MEDS batch.

        Args:
            batch: Input ``MEDSTorchBatch`` for the current minibatch.

        Returns:
            TaskTargets: Either a tensor target or a structured mapping of tensors.
        """

    @abstractmethod
    def metrics(self, predictions: ModelOutput, targets: TaskTargets) -> Mapping[str, Tensor]:
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
    def forward(self, predictions: ModelOutput, targets: TaskTargets) -> Tensor:
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

    Examples:
        >>> BinaryClassificationTask(output_dim=2)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: BinaryClassificationTask requires output_dim=1, got 2
    """

    def __init__(self, *, label_field: str = "boolean_value", output_dim: int = 1) -> None:
        super().__init__(output_dim=output_dim)
        if output_dim != 1:
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
            >>> singleton_batch = make_supervised_batch()
            >>> singleton_batch.boolean_value = torch.BoolTensor([[True], [False]])
            >>> tuple(task.extract_targets(singleton_batch).shape)
            (2,)
            >>> missing_targets = MEDSTorchBatch(
            ...     code=torch.LongTensor([[1, 2, 3], [3, 2, 1]]),
            ...     numeric_value=torch.zeros((2, 3), dtype=torch.float32),
            ...     numeric_value_mask=torch.zeros((2, 3), dtype=torch.bool),
            ...     time_delta_days=torch.zeros((2, 3), dtype=torch.float32),
            ... )
            >>> task.extract_targets(missing_targets)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: Expected boolean_value targets on the MEDS batch.
            >>> bad_batch = make_supervised_batch()
            >>> bad_batch.boolean_value = torch.BoolTensor([[True, False], [False, True]])
            >>> task.extract_targets(bad_batch)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: BinaryClassificationTask expects boolean_value shaped (B,) or (B, 1); got (2, 2)
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

    def metrics(self, predictions: ModelOutput, targets: TaskTargets) -> Mapping[str, Tensor]:
        """Return binary-accuracy metrics derived from logits.

        Args:
            predictions: Model predictions whose logits have shape ``(B, 1)``.
            targets: Binary targets with shape ``(B,)``.

        Returns:
            Mapping[str, Tensor]: Metric dictionary containing ``"accuracy"`` mapped
            to a scalar tensor with shape ``()``.

        Examples:
            >>> task = BinaryClassificationTask()
            >>> metrics = task.metrics(
            ...     ModelOutput(logits=torch.FloatTensor([[2.0], [-2.0]])),
            ...     torch.BoolTensor([True, False]),
            ... )
            >>> sorted(metrics)
            ['accuracy']
            >>> float(metrics["accuracy"])
            1.0
            >>> task.metrics(
            ...     ModelOutput(logits=torch.FloatTensor([2.0, -2.0])),
            ...     torch.BoolTensor([True, False]),
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: BinaryClassificationTask expects logits shaped (B, 1); got (2,)
        """
        flat_logits = _flatten_binary_logits(predictions, owner="BinaryClassificationTask")
        flat_targets = _flatten_binary_targets(targets, owner="BinaryClassificationTask").bool()
        predictions = flat_logits >= 0
        return {"accuracy": (predictions == flat_targets).float().mean()}


class MarginalizedBinaryClassificationTask(SupervisedTask):
    """Binary task with two logits per sample (marginal class distribution).

    Use with :class:`medrap.model.RetrievalAugmentedModel` when
    ``marginalized_retrieval=True`` and :class:`medrap.losses.MarginalizedRetrievalSupervisedLoss`.

    Examples:
        >>> import torch
        >>> from medrap.types import ModelOutput
        >>> MarginalizedBinaryClassificationTask(output_dim=3)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: MarginalizedBinaryClassificationTask requires output_dim=2, got 3
        >>> task = MarginalizedBinaryClassificationTask()
        >>> m = task.metrics(
        ...     ModelOutput(logits=torch.FloatTensor([[2.0, -1.0], [-1.0, 2.0]])),
        ...     torch.BoolTensor([True, False]),
        ... )
        >>> "accuracy" in m and m["accuracy"].shape == torch.Size([])
        True
        >>> tuple(task.extract_targets(make_supervised_batch()).shape)
        (2,)
        >>> mb = make_supervised_batch()
        >>> mb.boolean_value = torch.BoolTensor([[True], [False]])
        >>> tuple(task.extract_targets(mb).shape)
        (2,)
        >>> missing = MEDSTorchBatch(
        ...     code=torch.LongTensor([[1, 2], [3, 4]]),
        ...     numeric_value=torch.zeros((2, 2), dtype=torch.float32),
        ...     numeric_value_mask=torch.zeros((2, 2), dtype=torch.bool),
        ...     time_delta_days=torch.zeros((2, 2), dtype=torch.float32),
        ... )
        >>> task.extract_targets(missing)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Expected boolean_value targets...
        >>> bad = make_supervised_batch()
        >>> bad.boolean_value = torch.BoolTensor([[True, False], [False, True]])
        >>> task.extract_targets(bad)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: MarginalizedBinaryClassificationTask expects boolean_value shaped (B,) or (B, 1)...
        >>> task.metrics(
        ...     ModelOutput(logits=torch.zeros(2, 1)), torch.tensor([0.0, 1.0])
        ... )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: MarginalizedBinaryClassificationTask expects logits shaped (B, 2)...
    """

    def __init__(self, *, label_field: str = "boolean_value", output_dim: int = 2) -> None:
        super().__init__(output_dim=output_dim)
        if output_dim != 2:
            raise ValueError(f"MarginalizedBinaryClassificationTask requires output_dim=2, got {output_dim}")
        self.label_field = label_field

    def extract_targets(self, batch: MEDSTorchBatch) -> Tensor:
        """Same as :class:`BinaryClassificationTask` (float 0/1 targets)."""
        targets = getattr(batch, self.label_field, None)
        if not isinstance(targets, Tensor):
            raise ValueError(f"Expected {self.label_field} targets on the MEDS batch.")
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        elif targets.ndim != 1:
            raise ValueError(
                f"MarginalizedBinaryClassificationTask expects {self.label_field} shaped (B,) or (B, 1); "
                f"got {tuple(targets.shape)}"
            )
        return targets.float()

    def metrics(self, predictions: ModelOutput, targets: TaskTargets) -> Mapping[str, Tensor]:
        logits = predictions.logits
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(
                "MarginalizedBinaryClassificationTask expects logits shaped (B, 2); "
                f"got {tuple(logits.shape)}"
            )
        pred = logits.argmax(dim=-1)
        tgt = _flatten_binary_targets(targets, owner="MarginalizedBinaryClassificationTask").long()
        return {"accuracy": (pred == tgt).float().mean()}


class BinaryClassificationLoss(SupervisedLoss):
    """Binary BCE-with-logits loss for scalar binary predictions.

    Returns:
        BinaryClassificationLoss: Loss helper that accepts ``ModelOutput``
        predictions with logits shaped ``(B, 1)`` and binary tensor targets
        shaped ``(B,)``.
    """

    def forward(self, predictions: ModelOutput, targets: TaskTargets) -> Tensor:
        """Compute BCE-with-logits loss from binary predictions and targets.

        Args:
            predictions: ``ModelOutput`` predictions with logits shaped
                ``(B, 1)``.
            targets: Binary tensor targets shaped ``(B,)``.

        Returns:
            Tensor: Scalar loss tensor with shape ``()``.

        Examples:
            >>> loss_fn = BinaryClassificationLoss()
            >>> predictions = ModelOutput(logits=torch.FloatTensor([[0.0], [2.0]]))
            >>> targets = torch.BoolTensor([False, True])
            >>> round(float(loss_fn(predictions, targets)), 4)
            0.41
            >>> loss_fn(
            ...     ModelOutput(logits=torch.FloatTensor([[0.0], [2.0]])),
            ...     {"labels": torch.FloatTensor([0.0, 1.0])},
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: BinaryClassificationLoss expects tensor targets, not structured targets.
            >>> loss_fn(
            ...     ModelOutput(logits=torch.FloatTensor([[0.0], [2.0]])),
            ...     torch.BoolTensor([[False], [True]]),
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: BinaryClassificationLoss expects targets shaped (B,); got (2, 1)
        """
        flat_logits = _flatten_binary_logits(predictions, owner="BinaryClassificationLoss")
        flat_targets = _flatten_binary_targets(targets, owner="BinaryClassificationLoss")
        return torch.nn.functional.binary_cross_entropy_with_logits(flat_logits, flat_targets)
