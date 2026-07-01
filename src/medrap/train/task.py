"""Supervised task objects for training wrappers."""

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch  # noqa: F401 — needed in doctests (e.g. torch.float32)
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn

from ..types import ModelOutput

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


def _extract_boolean_value(batch: MEDSTorchBatch, label_field: str, owner: str) -> Tensor:
    targets = getattr(batch, label_field, None)
    if not isinstance(targets, Tensor):
        raise ValueError(f"Expected {label_field} targets on the MEDS batch.")
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    elif targets.ndim != 1:
        raise ValueError(f"{owner} expects {label_field} shaped (B,) or (B, 1); got {tuple(targets.shape)}")
    return targets.float()


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
        return _extract_boolean_value(batch, self.label_field, "BinaryClassificationTask")

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
        preds = flat_logits >= 0
        return {"accuracy": (preds == flat_targets).float().mean()}


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
        return _extract_boolean_value(batch, self.label_field, "MarginalizedBinaryClassificationTask")

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


class MultiTaskBinaryClassificationTask(SupervisedTask):
    """Multi-task binary classification: predict N binary outcomes simultaneously.

    Expects ``batch.multi_task_labels`` of shape ``(B, N)`` (float32, NaN = missing).
    The model must produce logits of shape ``(B, N)``.

    Args:
        num_tasks: Number of binary prediction tasks N.

    Examples:
        >>> import torch
        >>> from medrap.types import ModelOutput
        >>> task = MultiTaskBinaryClassificationTask(num_tasks=3)
        >>> task.output_dim
        3
        >>> MultiTaskBinaryClassificationTask(num_tasks=0)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: num_tasks must be a positive integer, got 0
    """

    def __init__(self, *, num_tasks: int) -> None:
        if num_tasks <= 0:
            raise ValueError(f"num_tasks must be a positive integer, got {num_tasks}")
        super().__init__(output_dim=num_tasks)
        self.num_tasks = num_tasks

    def extract_targets(self, batch: MEDSTorchBatch) -> Tensor:
        """Extract multi-task labels from the batch.

        Args:
            batch: ``MEDSTorchBatch`` with ``multi_task_labels`` attribute of shape ``(B, N)``.

        Returns:
            Float tensor of shape ``(B, N)`` with NaN for missing labels.

        Examples:
            >>> import torch
            >>> from meds_torchdata import MEDSTorchBatch
            >>> task = MultiTaskBinaryClassificationTask(num_tasks=2)
            >>> batch = MEDSTorchBatch(
            ...     code=torch.LongTensor([[1, 2], [3, 4]]),
            ...     numeric_value=torch.zeros(2, 2),
            ...     numeric_value_mask=torch.zeros(2, 2, dtype=torch.bool),
            ...     time_delta_days=torch.zeros(2, 2),
            ... )
            >>> batch.multi_task_labels = torch.tensor([[1.0, 0.0], [float("nan"), 1.0]])
            >>> targets = task.extract_targets(batch)
            >>> tuple(targets.shape)
            (2, 2)
            >>> targets.dtype
            torch.float32
            >>> no_labels = MEDSTorchBatch(
            ...     code=torch.LongTensor([[1, 2]]),
            ...     numeric_value=torch.zeros(1, 2),
            ...     numeric_value_mask=torch.zeros(1, 2, dtype=torch.bool),
            ...     time_delta_days=torch.zeros(1, 2),
            ... )
            >>> task.extract_targets(no_labels)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: Expected multi_task_labels on the MEDS batch.
        """
        labels = getattr(batch, "multi_task_labels", None)
        if not isinstance(labels, Tensor):
            raise ValueError("Expected multi_task_labels on the MEDS batch.")
        return labels.float()

    def metrics(self, predictions: ModelOutput, targets: TaskTargets) -> Mapping[str, Tensor]:
        """Return masked accuracy across all valid (patient, task) pairs.

        Examples:
            >>> import torch
            >>> from medrap.types import ModelOutput
            >>> task = MultiTaskBinaryClassificationTask(num_tasks=2)
            >>> preds = ModelOutput(logits=torch.tensor([[2.0, -1.0], [-1.0, 2.0]]))
            >>> targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            >>> float(task.metrics(preds, targets)["accuracy"])
            1.0
            >>> float(
            ...     task.metrics(
            ...         ModelOutput(logits=torch.tensor([[10.0, -10.0]])),
            ...         torch.tensor([[0.0, float("nan")]]),
            ...     )["accuracy"]
            ... )
            0.0
            >>> float(
            ...     task.metrics(ModelOutput(logits=torch.zeros(2, 2)), torch.full((2, 2), float("nan")))[
            ...         "accuracy"
            ...     ]
            ... )
            0.0
            >>> task.metrics(
            ...     ModelOutput(logits=torch.zeros(1, 2)), {"x": torch.zeros(1, 2)}
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: MultiTaskBinaryClassificationTask expects tensor targets.
        """
        if not isinstance(targets, Tensor):
            raise ValueError("MultiTaskBinaryClassificationTask expects tensor targets.")
        logits = predictions.logits  # (B, N)
        valid = ~targets.isnan()
        preds = (logits >= 0).float()
        correct = (preds == targets.nan_to_num(0.0)).float()
        return {"accuracy": (correct * valid.float()).sum() / valid.float().sum().clamp(min=1)}
