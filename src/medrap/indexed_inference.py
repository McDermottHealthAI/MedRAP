"""Shared orchestration for index-dataframe-driven inference commands.

``medrap.retrieve``, ``medrap.predict_probabilities``, and ``medrap.get_embeddings``
all share the same skeleton: given a trained model and an index dataframe naming a
set of MEDS patient-timepoints (``subject_id``/``prediction_time`` pairs, the same
"task index" concept ``meds_torchdata`` supports via
``MEDSTorchDataConfig(task_labels_dir=...)`` without a label column), run inference
over every patient-timepoint and save one artifact per row.

Because a patient-timepoint's subject belongs to exactly one MEDS split, inference
is run once per split in ``cfg.splits`` and the per-split results are concatenated,
then left-joined against the full input index dataframe so every input row is
represented in the output -- with null values for any patient-timepoint
``meds_torchdata`` could not build a sample for (e.g. a subject outside every
configured split, or insufficient history before its prediction time). Splits with
no shards in the tensorized cohort (e.g. no ``held_out`` split) are skipped rather
than treated as an error.

``run_indexed_inference`` is the shared entrypoint; callers supply an ``extract``
function mapping a split's collated ``predict_step`` output to the columns they
want attached to that split's ``(subject_id, prediction_time)`` rows.
"""

from collections.abc import Callable
from pathlib import Path

import lightning
import polars as pl
from meds_torchdata import MEDSPytorchDataset, MEDSTorchDataConfig
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader

from .extraction import collate_prediction_batches

Extractor = Callable[[dict[str, Tensor]], dict[str, list]]


def _build_dataset_config(cfg: DictConfig, *, task_labels_dir: str | Path) -> MEDSTorchDataConfig:
    """Build an inference ``MEDSTorchDataConfig`` from the training datamodule's dataset fields.

    Reuses ``cfg.training.datamodule.config``'s ``tensorized_cohort_dir``/
    ``max_seq_len``/``seq_sampling_strategy`` (the MEDS dataset the checkpoint was
    trained on must match), but points ``task_labels_dir`` at the inference index
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


def _predict_split(
    module: lightning.LightningModule,
    trainer: lightning.Trainer,
    dataset: MEDSPytorchDataset,
    *,
    batch_size: int,
    extract: Extractor,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> pl.DataFrame | None:
    """Run inference over one split's dataset and return a per-row result frame.

    Returns ``None`` when the split has no rows (e.g. the index dataframe has no
    subjects in this split). Otherwise returns a DataFrame with columns
    ``subject_id``, ``prediction_time``, plus whatever columns ``extract`` returns,
    in the same row order as ``dataset``.

    Args:
        module: Loaded Lightning module with a ``predict_step``.
        trainer: Single-device ``lightning.Trainer`` used for ``trainer.predict``.
        dataset: Split-scoped ``MEDSPytorchDataset``.
        batch_size: Prediction dataloader batch size.
        extract: Maps the split's collated ``predict_step`` output (a
            ``dict[str, Tensor]``, each tensor shaped ``(N, ...)``) to the output
            columns to attach, as ``dict[str, list]`` with per-row python values.
        num_workers: Prediction dataloader worker count.
        pin_memory: Whether to pin prediction dataloader memory.

    Raises:
        ValueError: If ``trainer`` uses more than one device -- a multi-device
            ``trainer.predict()`` is not guaranteed to return batches in dataset
            order, which would silently misalign results against
            ``dataset.schema_df`` (see ``medrap.extraction.extract_artifacts``,
            which guards against the same risk). ``extract`` may also raise if the
            collated predictions are missing columns it requires.

    Examples:
        >>> from types import SimpleNamespace
        >>> class _FakeDataset:
        ...     def __len__(self):
        ...         return 3
        >>> fake_trainer = SimpleNamespace(num_devices=2)
        >>> _predict_split(
        ...     None, fake_trainer, _FakeDataset(), batch_size=1, extract=lambda c: {}
        ... )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: medrap inference requires a single-device trainer...

        A split with no rows returns ``None`` without touching ``trainer``/``module``,
        so a split absent from the index dataframe is silently skipped by
        :func:`run_indexed_inference` rather than running an empty predict loop:

        >>> class _EmptyDataset:
        ...     def __len__(self):
        ...         return 0
        >>> _predict_split(None, None, _EmptyDataset(), batch_size=1, extract=lambda c: {}) is None
        True
    """
    if len(dataset) == 0:
        return None

    num_devices = getattr(trainer, "num_devices", 1)
    if num_devices and num_devices > 1:
        raise ValueError(
            "medrap inference requires a single-device trainer; multi-device predict "
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
    extracted = extract(collated)

    schema = dataset.schema_df.select("subject_id", "prediction_time")
    return schema.with_columns(**{name: pl.Series(name, values) for name, values in extracted.items()})


def run_indexed_inference(
    cfg: DictConfig,
    *,
    module: lightning.LightningModule,
    trainer: lightning.Trainer,
    extract: Extractor,
) -> pl.DataFrame:
    """Run inference for every patient-timepoint in ``cfg.index_dataframe_dir``.

    For each split in ``cfg.splits`` with a schema in the tensorized cohort, builds
    a ``MEDSPytorchDataset`` scoped to that split with
    ``task_labels_dir=cfg.index_dataframe_dir``, runs ``trainer.predict``, and calls
    ``extract`` on the collated results. Results across splits are concatenated (a
    patient-timepoint's subject belongs to exactly one split) and left-joined
    against the full input index dataframe.

    If every split ends up with no rows, the returned frame has only
    ``subject_id``/``prediction_time`` -- none of ``extract``'s columns, since there
    was no collated output to derive them from. Callers should add their own
    null-typed columns in that case before persisting a result with a fixed schema
    (see ``medrap.retrieve.retrieval.run_retrieval`` for the pattern).

    Args:
        cfg: Resolved inference config (``_retrieve``/``_predict_probabilities``/
            ``_get_embeddings``) with ``index_dataframe_dir``, ``splits``, and
            ``batch_size`` fields, plus ``training.datamodule`` for dataset
            construction.
        module: Loaded Lightning module (see ``_load_training_module_checkpoint``
            in ``medrap.cli``).
        trainer: Instantiated single-device ``lightning.Trainer``.
        extract: See :func:`_predict_split`.

    Raises:
        FileNotFoundError: If ``cfg.index_dataframe_dir`` contains no parquet files.

    Returns:
        Left-joined DataFrame, one row per input index-dataframe row.

    Examples:
        >>> import tempfile
        >>> from omegaconf import OmegaConf
        >>> with tempfile.TemporaryDirectory() as empty_dir:
        ...     cfg = OmegaConf.create({"index_dataframe_dir": empty_dir})
        ...     run_indexed_inference(
        ...         cfg, module=None, trainer=None, extract=lambda c: {}
        ...     )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        FileNotFoundError: No parquet files found under index_dataframe_dir=...
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
            extract=extract,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if frame is not None:
            per_split_frames.append(frame)

    results = pl.concat(per_split_frames, how="vertical") if per_split_frames else index_df.clear()

    return index_df.join(results, on=["subject_id", "prediction_time"], how="left")
