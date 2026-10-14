"""Build the long-format results table ``simple-scaling-laws`` expects from per-run records.

Column convention (see the ``simple-scaling-laws`` README): one row per evaluation
resample, ``training_run_id``/``train_set_id``/``test_set_id``/``optimizer_seed`` identify
the run, ``model_size__*``/``dataset_size__*`` are the scale predictors, and
``test_metric__*``/``test_loss__*`` are the fit targets.
"""

from collections.abc import Iterable
from typing import Any

import polars as pl
import torch.nn as nn


def encoder_param_count(encoder_cls: type[nn.Module], **encoder_kwargs: Any) -> int:
    """Exact trainable-parameter count for an encoder config, used as ``model_size__n_params``.

    Examples:
        >>> from medrap.model.encoders import TimeDeltaRoPEPatientEncoder
        >>> encoder_param_count(
        ...     TimeDeltaRoPEPatientEncoder,
        ...     vocab_size=65536,
        ...     embedding_dim=16,
        ...     num_heads=2,
        ...     num_layers=1,
        ...     ff_dim=32,
        ... )
        1050832
    """
    model = encoder_cls(**encoder_kwargs)
    return sum(p.numel() for p in model.parameters())


def build_results_table(records: Iterable[dict[str, Any]]) -> pl.DataFrame:
    """Reshape per-evaluation-resample run records into ``simple-scaling-laws`` input format.

    Each record must have: ``training_run_id``, ``train_set_id``, ``test_set_id``,
    ``optimizer_seed``, ``arm`` (the retrieval arm, used to namespace the loss column since
    different arms use different, not-directly-comparable loss functions),
    ``n_params``, ``n_subjects``, ``auroc``, ``loss``.

    Examples:
        >>> records = [
        ...     {
        ...         "training_run_id": "rope_dim16-frac0.1-patient_only-rep0",
        ...         "train_set_id": "frac0.1-rep0",
        ...         "test_set_id": "held_out",
        ...         "optimizer_seed": 0,
        ...         "arm": "patient_only",
        ...         "n_params": 1050832,
        ...         "n_subjects": 100,
        ...         "auroc": 0.61,
        ...         "loss": 0.68,
        ...     },
        ...     {
        ...         "training_run_id": "rope_dim16-frac0.1-retrieval-rep0",
        ...         "train_set_id": "frac0.1-rep0",
        ...         "test_set_id": "held_out",
        ...         "optimizer_seed": 0,
        ...         "arm": "retrieval",
        ...         "n_params": 1050832,
        ...         "n_subjects": 100,
        ...         "auroc": 0.67,
        ...         "loss": 0.55,
        ...     },
        ... ]
        >>> table = build_results_table(records)
        >>> sorted(c for c in table.columns if c.startswith(("test_loss__", "test_metric__")))
        ['test_loss__patient_only', 'test_loss__retrieval', 'test_metric__auroc']
        >>> retrieval_row = table.filter(pl.col("training_run_id").str.contains("retrieval"))
        >>> retrieval_row["test_loss__retrieval"][0]
        0.55
        >>> retrieval_row["test_loss__patient_only"][0] is None
        True
    """
    rows = list(records)
    arms = sorted({row["arm"] for row in rows})
    table_rows = []
    for row in rows:
        table_row = {
            "training_run_id": row["training_run_id"],
            "train_set_id": row["train_set_id"],
            "test_set_id": row["test_set_id"],
            "optimizer_seed": row["optimizer_seed"],
            "arm": row["arm"],
            "model_size__n_params": row["n_params"],
            "dataset_size__n_subjects": row["n_subjects"],
            "test_metric__auroc": row["auroc"],
        }
        for arm in arms:
            table_row[f"test_loss__{arm}"] = row["loss"] if row["arm"] == arm else None
        table_rows.append(table_row)
    return pl.DataFrame(table_rows)
