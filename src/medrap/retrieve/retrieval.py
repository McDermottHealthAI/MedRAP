"""Retrieval inference: run a trained model over patient-timepoints named by an index dataframe.

For each patient-timepoint (``subject_id``, ``prediction_time`` pair) in an index
dataframe, retrieves the configured retriever's top-K documents (or null, when
meds_torchdata cannot build a sample for that patient-timepoint -- e.g. the
subject falls outside every configured split, or has insufficient history) and
saves document IDs and scores to disk.

See ``medrap.indexed_inference`` for the shared per-split/left-join orchestration
this module builds on.
"""

from pathlib import Path

import lightning
import polars as pl
from omegaconf import DictConfig
from torch import Tensor

from ..indexed_inference import run_indexed_inference


def _squeeze_retrieval_step(tensor: Tensor, *, name: str) -> list:
    """Convert a ``(N, R, K)`` retrieval tensor to a per-row python list, requiring ``R == 1``.

    Sequence-mode retrieval (``R > 1``, multiple retrieval steps per
    patient-timepoint) is out of scope for ``medrap-retrieve`` v1, which targets
    the tabular/single-query-per-timepoint configurations used everywhere else in
    this codebase's default configs.

    Examples:
        >>> import torch
        >>> _squeeze_retrieval_step(torch.LongTensor([[[1, 2]], [[3, 4]]]), name="doc_ids")
        [[1, 2], [3, 4]]
        >>> _squeeze_retrieval_step(torch.zeros(2, 2, 3), name="doc_ids")
        Traceback (most recent call last):
            ...
        ValueError: medrap-retrieve only supports single-step retrieval (R=1); got doc_ids with R=2.
    """
    if tensor.shape[1] != 1:
        raise ValueError(
            f"medrap-retrieve only supports single-step retrieval (R=1); got {name} with R={tensor.shape[1]}."
        )
    return tensor[:, 0, :].tolist()


def _extract_retrieval(collated: dict[str, Tensor]) -> dict[str, list]:
    """Map a split's collated ``predict_step`` output to ``doc_ids``/``doc_scores`` columns.

    Examples:
        >>> import torch
        >>> _extract_retrieval(
        ...     {"doc_ids": torch.LongTensor([[[1, 2]]]), "doc_scores": torch.FloatTensor([[[0.5, 0.1]]])}
        ... )
        {'doc_ids': [[1, 2]], 'doc_scores': [[0.5, 0.10000000149011612]]}
        >>> _extract_retrieval({"logits": torch.zeros(1, 1)})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: medrap-retrieve requires a retriever that provides 'doc_ids'...
    """
    for key in ("doc_ids", "doc_scores"):
        if key not in collated:
            raise ValueError(
                f"medrap-retrieve requires a retriever that provides {key!r}; the configured "
                "retriever did not produce it."
            )
    return {
        "doc_ids": _squeeze_retrieval_step(collated["doc_ids"], name="doc_ids"),
        "doc_scores": _squeeze_retrieval_step(collated["doc_scores"], name="doc_scores"),
    }


def run_retrieval(cfg: DictConfig, *, module: lightning.LightningModule, trainer: lightning.Trainer) -> Path:
    """Retrieve documents for every patient-timepoint in ``cfg.index_dataframe_dir``.

    Delegates the per-split/left-join orchestration to
    ``medrap.indexed_inference.run_indexed_inference``, extracting ``doc_ids``/
    ``doc_scores`` (squeezed to ``list[K]``, requiring single-step retrieval) from
    each split's collated predictions.

    Args:
        cfg: Resolved ``_retrieve`` config.
        module: Loaded ``MedRAPSupervisedLightningModule`` (see
            ``_load_training_module_checkpoint`` in ``medrap.cli``).
        trainer: Instantiated ``lightning.Trainer`` used for ``trainer.predict``.

    Returns:
        Path to the written ``retrieved_documents.parquet``.
    """
    output = run_indexed_inference(cfg, module=module, trainer=trainer, extract=_extract_retrieval)

    if "doc_ids" not in output.columns:
        output = output.with_columns(
            pl.lit(None, dtype=pl.List(pl.Int64)).alias("doc_ids"),
            pl.lit(None, dtype=pl.List(pl.Float64)).alias("doc_scores"),
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "retrieved_documents.parquet"
    output.write_parquet(output_path)
    return output_path
