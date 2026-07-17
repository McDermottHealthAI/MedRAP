"""Prediction inference: run a trained model over patient-timepoints named by an index dataframe.

For each patient-timepoint (``subject_id``, ``prediction_time`` pair) in an index
dataframe, produces predicted probabilities for the training target labels (or
null, when meds_torchdata cannot build a sample for that patient-timepoint) and
saves them to disk.

See ``medrap.indexed_inference`` for the shared per-split/left-join orchestration
this module builds on.
"""

from pathlib import Path

import lightning
import polars as pl
import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from ..indexed_inference import run_indexed_inference
from ..train.metrics import positive_class_probs
from ..train.task import MultiTaskBinaryClassificationTask


def _predicted_probabilities(logits: Tensor, task: nn.Module) -> Tensor:
    """Convert task logits to predicted probabilities, shape ``(B, C)``.

    Dispatches on the *type* of ``task``, not the shape of ``logits`` alone --
    ``(B, 2)`` logits mean two independent binary tasks under
    ``MultiTaskBinaryClassificationTask`` (per-task sigmoid) but a single
    two-class marginal distribution under ``MarginalizedBinaryClassificationTask``
    (softmax positive-class probability), so shape alone can't disambiguate them.

    Args:
        logits: Model logits, ``(B, 1)`` for ``BinaryClassificationTask``,
            ``(B, 2)`` for ``MarginalizedBinaryClassificationTask``, or ``(B, N)``
            for ``MultiTaskBinaryClassificationTask``.
        task: The loaded module's task object.

    Returns:
        Tensor: Probabilities shaped ``(B, 1)`` for binary/marginalized tasks (the
        single positive-class probability), or ``(B, N)`` for multitask (one
        independent probability per task column).

    Examples:
        >>> from medrap.train.task import (
        ...     BinaryClassificationTask,
        ...     MarginalizedBinaryClassificationTask,
        ...     MultiTaskBinaryClassificationTask,
        ... )
        >>> _predicted_probabilities(torch.FloatTensor([[0.0], [2.0]]), BinaryClassificationTask()).round(
        ...     decimals=4
        ... )
        tensor([[0.5000],
                [0.8808]])
        >>> _predicted_probabilities(
        ...     torch.FloatTensor([[2.0, 0.0], [0.0, 2.0]]), MarginalizedBinaryClassificationTask()
        ... ).round(decimals=4)
        tensor([[0.1192],
                [0.8808]])
        >>> _predicted_probabilities(
        ...     torch.FloatTensor([[2.0, -2.0]]), MultiTaskBinaryClassificationTask(num_tasks=2)
        ... ).round(decimals=4)
        tensor([[0.8808, 0.1192]])
    """
    if isinstance(task, MultiTaskBinaryClassificationTask):
        return torch.sigmoid(logits)
    return positive_class_probs(logits).unsqueeze(-1)


def _extract_probabilities(module: nn.Module):
    """Return an ``extract`` closure mapping collated logits to a ``probabilities`` column.

    Examples:
        >>> import torch
        >>> from medrap.train.task import BinaryClassificationTask
        >>> from types import SimpleNamespace
        >>> fake_module = SimpleNamespace(task=BinaryClassificationTask())
        >>> extract = _extract_probabilities(fake_module)
        >>> extract({"logits": torch.FloatTensor([[0.0], [2.0]])})["probabilities"]
        [[0.5], [0.8807970285415649]]
    """

    def extract(collated: dict[str, Tensor]) -> dict[str, list]:
        probabilities = _predicted_probabilities(collated["logits"], module.task)
        return {"probabilities": probabilities.tolist()}

    return extract


def run_predict_probabilities(
    cfg: DictConfig, *, module: lightning.LightningModule, trainer: lightning.Trainer
) -> Path:
    """Predict probabilities for every patient-timepoint in ``cfg.index_dataframe_dir``.

    Delegates the per-split/left-join orchestration to
    ``medrap.indexed_inference.run_indexed_inference``, extracting a
    ``probabilities`` (``list[C]``) column from each split's collated ``logits``,
    task-aware per :func:`_predicted_probabilities`.

    Args:
        cfg: Resolved ``_predict_probabilities`` config.
        module: Loaded ``MedRAPSupervisedLightningModule`` (see
            ``_load_training_module_checkpoint`` in ``medrap.cli``).
        trainer: Instantiated ``lightning.Trainer`` used for ``trainer.predict``.

    Returns:
        Path to the written ``probabilities.parquet``.
    """
    output = run_indexed_inference(
        cfg, module=module, trainer=trainer, extract=_extract_probabilities(module)
    )

    if "probabilities" not in output.columns:
        output = output.with_columns(pl.lit(None, dtype=pl.List(pl.Float64)).alias("probabilities"))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "probabilities.parquet"
    output.write_parquet(output_path)
    return output_path
