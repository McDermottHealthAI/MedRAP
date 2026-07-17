"""Retrieval inference: run a trained model over patient-timepoints named by an index dataframe.

For each patient-timepoint (``subject_id``, ``prediction_time`` pair) in an index
dataframe, retrieves the configured retriever's top-K documents (or null, when
meds_torchdata cannot build a sample for that patient-timepoint -- e.g. the
subject falls outside every configured split, or has insufficient history) and
saves document IDs and scores to disk.

The index dataframe is a directory of parquet file(s) with ``subject_id`` and
``prediction_time`` columns -- the same "task index" concept meds_torchdata
already supports via ``MEDSTorchDataConfig(task_labels_dir=...)`` without a label
column. Because a patient-timepoint's subject belongs to exactly one MEDS split,
retrieval is run once per split in ``cfg.splits`` and the per-split results are
concatenated, then left-joined against the full input index dataframe so every
input row is represented in the output.
"""

from pathlib import Path

import lightning
import polars as pl
from meds_torchdata import MEDSPytorchDataset, MEDSTorchDataConfig
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader

from ..extraction import collate_prediction_batches


def _build_dataset_config(cfg: DictConfig, *, task_labels_dir: str | Path) -> MEDSTorchDataConfig:
    """Build a retrieval ``MEDSTorchDataConfig`` from the training datamodule's dataset fields.

    Reuses ``cfg.training.datamodule.config``'s ``tensorized_cohort_dir``/
    ``max_seq_len``/``seq_sampling_strategy`` (the MEDS dataset the checkpoint was
    trained on must match), but points ``task_labels_dir`` at the retrieve index
    dataframe rather than whatever training labels ``cfg`` was composed with.

    Examples:
        >>> import tempfile
        >>> from hydra import compose, initialize_config_module
        >>> cohort_dir = tempfile.mkdtemp()
        >>> index_dir = tempfile.mkdtemp()
        >>> with initialize_config_module(version_base=None, config_module="medrap.conf"):
        ...     cfg = compose(
        ...         config_name="_retrieve",
        ...         overrides=[
        ...             "training/datamodule=meds",
        ...             "output_dir=outputs/demo",
        ...             "checkpoint_path=outputs/demo/checkpoints/last.ckpt",
        ...             f"index_dataframe_dir={index_dir}",
        ...             f"training.datamodule.config.tensorized_cohort_dir={cohort_dir}",
        ...             "training.datamodule.config.max_seq_len=8",
        ...         ],
        ...     )
        >>> dataset_config = _build_dataset_config(cfg, task_labels_dir=index_dir)
        >>> dataset_config.max_seq_len
        8
        >>> str(dataset_config.task_labels_dir) == index_dir
        True
    """
    dm_config = cfg.training.datamodule.config
    return MEDSTorchDataConfig(
        tensorized_cohort_dir=dm_config.tensorized_cohort_dir,
        max_seq_len=dm_config.max_seq_len,
        task_labels_dir=str(task_labels_dir),
        seq_sampling_strategy=dm_config.get("seq_sampling_strategy", "to_end"),
    )


def _split_has_schema(dataset_config: MEDSTorchDataConfig, split: str) -> bool:
    """Return whether ``dataset_config``'s tensorized cohort has any shard for ``split``.

    Mirrors the check ``MEDSPytorchDataset.__init__`` performs internally; lets
    callers skip a split with no shards (e.g. a cohort with no ``held_out`` split)
    instead of hitting the ``FileNotFoundError`` it raises when no shard matches.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     schema_dir = Path(tmpdir) / "tokenization" / "schemas" / "train"
        ...     schema_dir.mkdir(parents=True)
        ...     _ = (schema_dir / "0.parquet").touch()
        ...     config = MEDSTorchDataConfig(tensorized_cohort_dir=tmpdir, max_seq_len=8)
        ...     _split_has_schema(config, "train")
        ...     _split_has_schema(config, "held_out")
        True
        False
    """
    return any(shard.startswith(f"{split}/") for shard, _ in dataset_config.schema_fps)


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


def _predict_split(
    module: lightning.LightningModule,
    trainer: lightning.Trainer,
    dataset: MEDSPytorchDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> pl.DataFrame | None:
    """Run retrieval prediction over one split's dataset and return a per-row result frame.

    Returns ``None`` when the split has no rows (e.g. the index dataframe has no
    subjects in this split). Otherwise returns a DataFrame with columns
    ``subject_id``, ``prediction_time``, ``doc_ids`` (``list[K]``), ``doc_scores``
    (``list[K]``), in the same row order as ``dataset``.

    Raises:
        ValueError: If the configured retriever does not provide ``doc_ids`` or
            ``doc_scores``, or if ``trainer`` uses more than one device -- a
            multi-device ``trainer.predict()`` is not guaranteed to return batches
            in dataset order, which would silently misalign results against
            ``dataset.schema_df`` (see ``medrap.extraction.extract_artifacts``,
            which guards against the same risk).

    Examples:
        >>> from types import SimpleNamespace
        >>> class _FakeDataset:
        ...     def __len__(self):
        ...         return 3
        >>> fake_trainer = SimpleNamespace(num_devices=2)
        >>> _predict_split(None, fake_trainer, _FakeDataset(), batch_size=1)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: medrap-retrieve requires a single-device trainer...
    """
    if len(dataset) == 0:
        return None

    num_devices = getattr(trainer, "num_devices", 1)
    if num_devices and num_devices > 1:
        raise ValueError(
            "medrap-retrieve requires a single-device trainer; multi-device predict "
            f"can return rank-interleaved outputs (got num_devices={num_devices})."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    predictions = trainer.predict(module, dataloaders=dataloader)
    collated = collate_prediction_batches(predictions)

    for key in ("doc_ids", "doc_scores"):
        if key not in collated:
            raise ValueError(
                f"medrap-retrieve requires a retriever that provides {key!r}; the configured "
                "retriever did not produce it."
            )

    schema = dataset.schema_df.select("subject_id", "prediction_time")
    return schema.with_columns(
        pl.Series("doc_ids", _squeeze_retrieval_step(collated["doc_ids"], name="doc_ids")),
        pl.Series("doc_scores", _squeeze_retrieval_step(collated["doc_scores"], name="doc_scores")),
    )


def run_retrieval(cfg: DictConfig, *, module: lightning.LightningModule, trainer: lightning.Trainer) -> Path:
    """Retrieve documents for every patient-timepoint in ``cfg.index_dataframe_dir``.

    For each split in ``cfg.splits``, builds a ``MEDSPytorchDataset`` scoped to
    that split with ``task_labels_dir=cfg.index_dataframe_dir``, runs
    ``trainer.predict``, and collects ``doc_ids``/``doc_scores`` per
    patient-timepoint. Results across splits are concatenated (a
    patient-timepoint's subject belongs to exactly one split) and left-joined
    against the full input index dataframe, so every input row is represented in
    the output -- with null ``doc_ids``/``doc_scores`` for any patient-timepoint
    meds_torchdata could not build a sample for (e.g. a subject outside every
    configured split, or insufficient history before its prediction time).

    Args:
        cfg: Resolved ``_retrieve`` config.
        module: Loaded ``MedRAPSupervisedLightningModule`` (see
            ``_load_training_module_checkpoint`` in ``medrap.cli``).
        trainer: Instantiated ``lightning.Trainer`` used for ``trainer.predict``.

    Returns:
        Path to the written ``retrieved_documents.parquet``.
    """
    index_dataframe_dir = Path(cfg.index_dataframe_dir)
    index_files = sorted(index_dataframe_dir.rglob("*.parquet"))
    if not index_files:
        raise FileNotFoundError(f"No parquet files found under index_dataframe_dir={index_dataframe_dir}.")
    index_df = pl.concat(
        [pl.read_parquet(fp, columns=["subject_id", "prediction_time"]) for fp in index_files],
        how="vertical",
    )

    dataset_config = _build_dataset_config(cfg, task_labels_dir=index_dataframe_dir)
    dm_cfg = cfg.training.datamodule
    num_workers = dm_cfg.get("num_workers", None) or 0
    pin_memory = bool(dm_cfg.get("pin_memory", None) or False)

    per_split_frames = []
    for split in cfg.splits:
        if not _split_has_schema(dataset_config, split):
            continue
        dataset = MEDSPytorchDataset(dataset_config, split=split)
        frame = _predict_split(
            module,
            trainer,
            dataset,
            batch_size=cfg.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if frame is not None:
            per_split_frames.append(frame)

    results = (
        pl.concat(per_split_frames, how="vertical")
        if per_split_frames
        else index_df.clear().with_columns(
            pl.lit(None, dtype=pl.List(pl.Int64)).alias("doc_ids"),
            pl.lit(None, dtype=pl.List(pl.Float64)).alias("doc_scores"),
        )
    )

    output = index_df.join(results, on=["subject_id", "prediction_time"], how="left")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "retrieved_documents.parquet"
    output.write_parquet(output_path)
    return output_path
