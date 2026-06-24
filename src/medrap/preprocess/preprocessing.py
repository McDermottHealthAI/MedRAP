"""Offline preprocessing for raw MEDS datasets: rare-code and sparse-subject filtering.

Runs before tensorization (``meds_torchdata``'s ``MTD_preprocess``), on the raw
MEDS directory, writing a filtered dataset in the same raw-MEDS shape.
"""

import tempfile
from pathlib import Path

import polars as pl
from MEDS_transforms.stages.filter_measurements.filter_measurements import filter_measurements
from MEDS_transforms.stages.filter_subjects.filter_subjects import filter_subjects
from omegaconf import OmegaConf

# Large enough to always pass any realistic min_subjects_per_code/min_occurrences_per_code
# threshold, used to exempt sentinel codes from frequency filtering below.
_ALWAYS_PASS_THRESHOLD = 2**31


def aggregate_code_metadata_from_meds(meds_data_dir: str | Path) -> pl.DataFrame:
    """Compute per-code subject/occurrence counts across all shards of a raw MEDS dataset.

    Args:
        meds_data_dir: Root of a raw MEDS dataset (containing a ``data/<split>/*.parquet`` tree).

    Returns:
        DataFrame with columns ``code``, ``code/n_subjects``, ``code/n_occurrences``.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {"subject_id": [1, 1, 2], "code": ["A", "B", "A"], "time": [1, 2, 1]}
        ...     ).write_parquet(shard_dir / "0.parquet")
        ...     result = aggregate_code_metadata_from_meds(tmpdir).sort("code")
        ...     result.select("code", "code/n_subjects", "code/n_occurrences").rows()
        [('A', 2, 2), ('B', 1, 1)]
    """
    data = pl.scan_parquet(Path(meds_data_dir) / "data" / "*" / "*.parquet")
    return (
        data.group_by("code")
        .agg(
            pl.col("subject_id").n_unique().alias("code/n_subjects"),
            pl.len().alias("code/n_occurrences"),
        )
        .collect()
    )


def _write_filtered_shards(
    meds_data_dir: str | Path,
    output_dir: str | Path,
    filter_fn: "pl.LazyFrame | None",
) -> Path:
    in_dir = Path(meds_data_dir) / "data"
    out_dir = Path(output_dir) / "data"
    for shard_path in sorted(in_dir.glob("*/*.parquet")):
        split_dir = out_dir / shard_path.parent.name
        split_dir.mkdir(parents=True, exist_ok=True)
        filtered = filter_fn(pl.scan_parquet(shard_path))
        filtered.collect().write_parquet(split_dir / shard_path.name)
    return Path(output_dir)


def filter_rare_codes(
    meds_data_dir: str | Path,
    code_metadata: pl.DataFrame,
    *,
    min_subjects_per_code: int | None,
    min_occurrences_per_code: int | None,
    sentinel_code_regex: str,
    output_dir: str | Path,
) -> Path:
    """Drop dynamic codes below frequency thresholds, exempting ``sentinel_code_regex`` matches.

    Writes the filtered dataset to ``output_dir`` in the same raw-MEDS
    ``data/<split>/<shard>.parquet`` shape as the input.

    Args:
        meds_data_dir: Root of a raw MEDS dataset.
        code_metadata: Output of :func:`aggregate_code_metadata_from_meds`.
        min_subjects_per_code: Minimum distinct subjects a code must appear in
            to be kept, or ``None`` for no threshold.
        min_occurrences_per_code: Minimum total occurrences a code must have
            to be kept, or ``None`` for no threshold.
        sentinel_code_regex: Codes matching this regex are kept regardless of
            frequency (e.g. death/admission/discharge events).
        output_dir: Directory to write the filtered dataset into.

    Returns:
        ``output_dir`` as a ``Path``.

    Examples:
        A rare sentinel-matching code survives filtering; a rare non-sentinel
        code does not:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1, 1, 2, 2],
        ...             "code": ["common", "rare", "common", "MEDS_DEATH"],
        ...             "time": [1, 2, 1, 2],
        ...         }
        ...     ).write_parquet(shard_dir / "0.parquet")
        ...     metadata = aggregate_code_metadata_from_meds(tmpdir)
        ...     out_dir = filter_rare_codes(
        ...         tmpdir,
        ...         metadata,
        ...         min_subjects_per_code=2,
        ...         min_occurrences_per_code=None,
        ...         sentinel_code_regex="MEDS_DEATH.*",
        ...         output_dir=Path(tmpdir) / "filtered",
        ...     )
        ...     sorted(pl.read_parquet(out_dir / "data" / "train" / "0.parquet")["code"].to_list())
        ['MEDS_DEATH', 'common', 'common']
    """
    is_sentinel = pl.col("code").str.contains(sentinel_code_regex)
    adjusted_metadata = code_metadata.with_columns(
        pl.when(is_sentinel).then(_ALWAYS_PASS_THRESHOLD).otherwise(pl.col("code/n_subjects")).alias(
            "code/n_subjects"
        ),
        pl.when(is_sentinel)
        .then(_ALWAYS_PASS_THRESHOLD)
        .otherwise(pl.col("code/n_occurrences"))
        .alias("code/n_occurrences"),
    )
    stage_cfg = OmegaConf.create(
        {
            "min_subjects_per_code": min_subjects_per_code,
            "min_occurrences_per_code": min_occurrences_per_code,
        }
    )
    filter_fn = filter_measurements(stage_cfg, adjusted_metadata)
    return _write_filtered_shards(meds_data_dir, output_dir, filter_fn)


def filter_sparse_subjects(
    meds_data_dir: str | Path,
    *,
    min_events_per_subject: int | None,
    output_dir: str | Path,
) -> Path:
    """Drop subjects with fewer than ``min_events_per_subject`` distinct event timepoints.

    Writes the filtered dataset to ``output_dir`` in the same raw-MEDS
    ``data/<split>/<shard>.parquet`` shape as the input.

    Args:
        meds_data_dir: Root of a raw MEDS dataset.
        min_events_per_subject: Minimum distinct event timepoints a subject
            must have to be kept, or ``None``/``0`` for no threshold.
        output_dir: Directory to write the filtered dataset into.

    Returns:
        ``output_dir`` as a ``Path``.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1, 1, 2],
        ...             "code": ["A", "B", "A"],
        ...             "time": [1, 2, 1],
        ...         }
        ...     ).write_parquet(shard_dir / "0.parquet")
        ...     out_dir = filter_sparse_subjects(
        ...         tmpdir, min_events_per_subject=2, output_dir=Path(tmpdir) / "filtered"
        ...     )
        ...     pl.read_parquet(out_dir / "data" / "train" / "0.parquet")["subject_id"].to_list()
        [1, 1]
    """
    stage_cfg = OmegaConf.create(
        {
            "min_measurements_per_subject": None,
            "min_events_per_subject": min_events_per_subject,
        }
    )
    filter_fn = filter_subjects(stage_cfg)
    return _write_filtered_shards(meds_data_dir, output_dir, filter_fn)


def preprocess_meds_dataset(
    *,
    meds_data_dir: str | Path,
    output_dir: str | Path,
    min_subjects_per_code: int | None,
    min_occurrences_per_code: int | None,
    sentinel_code_regex: str,
    min_events_per_subject: int | None,
) -> Path:
    """Filter rare codes then sparse subjects from a raw MEDS dataset.

    Args:
        meds_data_dir: Root of a raw MEDS dataset.
        output_dir: Directory to write the filtered dataset into.
        min_subjects_per_code: See :func:`filter_rare_codes`.
        min_occurrences_per_code: See :func:`filter_rare_codes`.
        sentinel_code_regex: See :func:`filter_rare_codes`.
        min_events_per_subject: See :func:`filter_sparse_subjects`.

    Returns:
        ``output_dir`` as a ``Path``.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "raw" / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1, 1, 2],
        ...             "code": ["common", "rare", "common"],
        ...             "time": [1, 2, 1],
        ...         }
        ...     ).write_parquet(shard_dir / "0.parquet")
        ...     out_dir = preprocess_meds_dataset(
        ...         meds_data_dir=Path(tmpdir) / "raw",
        ...         output_dir=Path(tmpdir) / "filtered",
        ...         min_subjects_per_code=2,
        ...         min_occurrences_per_code=None,
        ...         sentinel_code_regex="MEDS_DEATH.*",
        ...         min_events_per_subject=None,
        ...     )
        ...     sorted(pl.read_parquet(out_dir / "data" / "train" / "0.parquet")["code"].to_list())
        ['common', 'common']
    """
    code_metadata = aggregate_code_metadata_from_meds(meds_data_dir)
    with tempfile.TemporaryDirectory() as code_filtered_dir:
        filter_rare_codes(
            meds_data_dir,
            code_metadata,
            min_subjects_per_code=min_subjects_per_code,
            min_occurrences_per_code=min_occurrences_per_code,
            sentinel_code_regex=sentinel_code_regex,
            output_dir=code_filtered_dir,
        )
        return filter_sparse_subjects(
            code_filtered_dir,
            min_events_per_subject=min_events_per_subject,
            output_dir=output_dir,
        )
