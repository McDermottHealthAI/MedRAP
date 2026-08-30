"""Embedding extraction: run a trained model over patient-timepoints named by an index dataframe.

For each patient-timepoint (``subject_id``, ``prediction_time`` pair) in an index
dataframe, extracts the query embedding produced by the trained model's
encoder/query-projector stage (or null, when meds_torchdata cannot build a sample
for that patient-timepoint) and saves it to disk.

Extracting the model's final hidden-layer (post-fusion/pooling) representation
instead of the query embedding is out of scope for v1 -- ``predict_step`` does not
currently surface pooled/fused state, only the query-projection and retrieval
stages (see ``medrap.train.lightning_module.MedRAPSupervisedLightningModule.predict_step``).

See ``medrap.indexed_inference`` for the shared per-split/left-join orchestration
this module builds on.
"""

from pathlib import Path

import lightning
import polars as pl
from omegaconf import DictConfig
from torch import Tensor

from ..indexed_inference import run_indexed_inference


def _extract_embeddings(collated: dict[str, Tensor]) -> dict[str, list]:
    """Map a split's collated ``predict_step`` output to an ``embedding`` column.

    Squeezes the ``(N, R, D_ret)`` ``query_embeddings`` tensor to ``(N, D_ret)``,
    requiring single-step retrieval (``R == 1``) -- multi-step sequence-mode
    retrieval is out of scope for ``medrap-get-embeddings`` v1, matching the same
    scope boundary ``medrap.retrieve.retrieval`` draws for ``doc_ids``/``doc_scores``.

    Examples:
        >>> import torch
        >>> _extract_embeddings({"query_embeddings": torch.FloatTensor([[[1.0, 2.0]], [[3.0, 4.0]]])})
        {'embedding': [[1.0, 2.0], [3.0, 4.0]]}
        >>> _extract_embeddings({"logits": torch.zeros(1, 1)})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: medrap-get-embeddings requires query_embeddings in predict_step's output...
        >>> _extract_embeddings({"query_embeddings": torch.zeros(2, 2, 4)})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: medrap-get-embeddings only supports single-step retrieval (R=1); got R=2.
    """
    if "query_embeddings" not in collated:
        raise ValueError(
            "medrap-get-embeddings requires query_embeddings in predict_step's output; "
            "the configured query projector did not produce it."
        )
    query_embeddings = collated["query_embeddings"]
    if query_embeddings.shape[1] != 1:
        raise ValueError(
            "medrap-get-embeddings only supports single-step retrieval (R=1); "
            f"got R={query_embeddings.shape[1]}."
        )
    return {"embedding": query_embeddings[:, 0, :].tolist()}


def run_get_embeddings(
    cfg: DictConfig, *, module: lightning.LightningModule, trainer: lightning.Trainer
) -> Path:
    """Extract query embeddings for every patient-timepoint in ``cfg.index_dataframe_dir``.

    Delegates the per-split/left-join orchestration to
    ``medrap.indexed_inference.run_indexed_inference``, extracting an ``embedding``
    (``list[D_ret]``) column from each split's collated ``query_embeddings``.

    Args:
        cfg: Resolved ``_get_embeddings`` config.
        module: Loaded ``MedRAPSupervisedLightningModule`` (see
            ``_load_training_module_checkpoint`` in ``medrap.cli``).
        trainer: Instantiated ``lightning.Trainer`` used for ``trainer.predict``.

    Returns:
        Path to the written ``embeddings.parquet``.
    """
    output = run_indexed_inference(cfg, module=module, trainer=trainer, extract=_extract_embeddings)

    if "embedding" not in output.columns:
        output = output.with_columns(pl.lit(None, dtype=pl.List(pl.Float64)).alias("embedding"))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "embeddings.parquet"
    output.write_parquet(output_path)
    return output_path
