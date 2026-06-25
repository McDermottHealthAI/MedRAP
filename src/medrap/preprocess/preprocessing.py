"""Offline preprocessing for raw MEDS datasets: rare-code and sparse-subject filtering.

Runs before tensorization (``meds_torchdata``'s ``MTD_preprocess``), on the raw
MEDS directory, writing a filtered dataset in the same raw-MEDS shape.
"""

import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import polars as pl
from MEDS_transforms.stages.bin_numeric_values.bin_numeric_values import bin_numeric_values_fntr
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
    filter_fn: Callable[[pl.LazyFrame], pl.LazyFrame],
) -> Path:
    in_dir = Path(meds_data_dir) / "data"
    out_dir = Path(output_dir) / "data"
    for shard_path in sorted(in_dir.glob("*/*.parquet")):
        split_dir = out_dir / shard_path.parent.name
        split_dir.mkdir(parents=True, exist_ok=True)
        filtered = filter_fn(pl.scan_parquet(shard_path))
        filtered.collect().write_parquet(split_dir / shard_path.name)
    return Path(output_dir)


def fit_quantile_metadata(meds_data_dir: str | Path, n_bins: int) -> pl.DataFrame:
    """Compute per-code quantile breakpoints from the training split for numeric value binning.

    Only scans the training split (``data/train/``) to avoid leaking tuning or
    held-out distributional information into the bin boundaries. Codes with no
    non-null numeric values in the training split are omitted from the result;
    they will not be binned when the returned metadata is passed to
    ``bin_numeric_values_fntr``.

    Args:
        meds_data_dir: Root of a raw MEDS dataset.
        n_bins: Number of quantile bins to create. Produces ``n_bins - 1``
            evenly-spaced quantile breakpoints at levels ``1/n_bins``,
            ``2/n_bins``, …, ``(n_bins-1)/n_bins``.

    Returns:
        DataFrame with columns ``code`` and ``values/quantiles`` (a struct
        with field names ``values/quantile/{level}`` for each breakpoint level),
        in the format expected by ``MEDS_transforms``' ``bin_numeric_values_fntr``.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     train_dir = Path(tmpdir) / "data" / "train"
        ...     train_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1, 1, 2, 2, 3],
        ...             "code": ["Lab//A", "Lab//A", "Lab//A", "Lab//B", "NoValue"],
        ...             "numeric_value": pl.Series([1.0, 3.0, 5.0, 2.0, None], dtype=pl.Float32),
        ...             "time": pl.Series([None] * 5, dtype=pl.Datetime("us")),
        ...         }
        ...     ).write_parquet(train_dir / "0.parquet")
        ...     meta = fit_quantile_metadata(tmpdir, n_bins=2)
        ...     # NoValue has no numeric values so it's omitted; both quantile codes present
        ...     sorted(meta["code"].to_list())
        ['Lab//A', 'Lab//B']
    """
    quantile_levels = [round(i / n_bins, 10) for i in range(1, n_bins)]
    df = pl.scan_parquet(Path(meds_data_dir) / "data" / "train" / "*.parquet")

    nv_dtype = df.collect_schema().get("numeric_value", pl.Float32)

    quant_aggs = [
        pl.col("numeric_value").quantile(level).cast(nv_dtype).alias(f"values/quantile/{level}")
        for level in quantile_levels
    ]
    quant_col_names = [f"values/quantile/{level}" for level in quantile_levels]

    return (
        df.filter(pl.col("numeric_value").is_not_null())
        .group_by("code")
        .agg(*quant_aggs)
        .with_columns(pl.struct(quant_col_names).alias("values/quantiles"))
        .drop(quant_col_names)
        .collect()
    )


def quantize_numeric_values(
    meds_data_dir: str | Path,
    code_metadata: pl.DataFrame,
    *,
    output_dir: str | Path,
) -> Path:
    """Bin numeric values into quantile ranges, appending the range to the code name.

    Each measurement with a non-null numeric value whose code appears in
    ``code_metadata`` gets its code suffixed with the bin range (e.g.
    ``Creatinine//value_[0.9,1.3)``), and its ``numeric_value`` is set to null
    (the information is now encoded in the code name). Codes with no entry in
    ``code_metadata`` or with a null numeric value are left unchanged.

    Writes the result to ``output_dir`` in the same raw-MEDS
    ``data/<split>/<shard>.parquet`` shape as the input.

    Args:
        meds_data_dir: Root of a raw MEDS dataset.
        code_metadata: Output of :func:`fit_quantile_metadata`.
        output_dir: Directory to write the quantized dataset into.

    Returns:
        ``output_dir`` as a ``Path``.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     train_dir = Path(tmpdir) / "data" / "train"
        ...     train_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1, 2, 3, 1],
        ...             "code": ["Lab//A", "Lab//A", "Lab//A", "GENDER//F"],
        ...             "numeric_value": pl.Series([1.0, 3.0, 5.0, None], dtype=pl.Float32),
        ...             "time": pl.Series([None] * 4, dtype=pl.Datetime("us")),
        ...         }
        ...     ).write_parquet(train_dir / "0.parquet")
        ...     meta = fit_quantile_metadata(tmpdir, n_bins=2)
        ...     out_dir = quantize_numeric_values(tmpdir, meta, output_dir=Path(tmpdir) / "binned")
        ...     result = pl.read_parquet(out_dir / "data" / "train" / "0.parquet")
        ...     sorted(result["code"].to_list())
        ['GENDER//F', 'Lab//A//value_[-inf,3.0)', 'Lab//A//value_[3.0,inf)', 'Lab//A//value_[3.0,inf)']
        >>> result["numeric_value"].is_null().all()
        True
    """
    stage_cfg = OmegaConf.create({"drop_numeric_value": True})
    bin_fn = bin_numeric_values_fntr(stage_cfg, code_metadata)
    return _write_filtered_shards(meds_data_dir, output_dir, bin_fn)


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
        pl.when(is_sentinel)
        .then(_ALWAYS_PASS_THRESHOLD)
        .otherwise(pl.col("code/n_subjects"))
        .alias("code/n_subjects"),
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
    n_quantile_bins: int | None,
    min_subjects_per_code: int | None,
    min_occurrences_per_code: int | None,
    sentinel_code_regex: str,
    min_events_per_subject: int | None,
) -> Path:
    """Filter rare codes, quantize numeric values, then filter sparse subjects.

    Pipeline order (each stage writes to a temp directory feeding the next):

    1. **Filter rare codes**: drop codes below ``min_subjects_per_code`` /
       ``min_occurrences_per_code``, exempting ``sentinel_code_regex`` matches.
    2. **Quantize** (if ``n_quantile_bins`` is not ``None``): bin each surviving
       code's ``numeric_value`` into a quantile range and append the range to the
       code name (e.g. ``Creatinine//value_[0.9,1.3)``). Bin boundaries are fit
       on the training split of the filtered dataset to avoid leakage.
    3. **Filter sparse subjects**: drop subjects with fewer than
       ``min_events_per_subject`` distinct event timepoints.

    Args:
        meds_data_dir: Root of a raw MEDS dataset.
        output_dir: Directory to write the filtered dataset into.
        n_quantile_bins: Number of quantile bins for numeric value quantization,
            or ``None`` to skip. See :func:`fit_quantile_metadata`.
        min_subjects_per_code: See :func:`filter_rare_codes`.
        min_occurrences_per_code: See :func:`filter_rare_codes`.
        sentinel_code_regex: See :func:`filter_rare_codes`.
        min_events_per_subject: See :func:`filter_sparse_subjects`.

    Returns:
        ``output_dir`` as a ``Path``.

    Examples:
        Without quantization (existing behaviour):

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
        ...         n_quantile_bins=None,
        ...         min_subjects_per_code=2,
        ...         min_occurrences_per_code=None,
        ...         sentinel_code_regex="MEDS_DEATH.*",
        ...         min_events_per_subject=None,
        ...     )
        ...     sorted(pl.read_parquet(out_dir / "data" / "train" / "0.parquet")["code"].to_list())
        ['common', 'common']

        With quantization — ``GENDER//F`` (1 subject) is filtered first; then Lab//A's
        3 surviving values are binned, producing two codes, both kept:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "raw" / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1, 2, 3, 1],
        ...             "code": ["Lab//A", "Lab//A", "Lab//A", "GENDER//F"],
        ...             "numeric_value": pl.Series([1.0, 3.0, 5.0, None], dtype=pl.Float32),
        ...             "time": pl.Series([None] * 4, dtype=pl.Datetime("us")),
        ...         }
        ...     ).write_parquet(shard_dir / "0.parquet")
        ...     out_dir = preprocess_meds_dataset(
        ...         meds_data_dir=Path(tmpdir) / "raw",
        ...         output_dir=Path(tmpdir) / "filtered",
        ...         n_quantile_bins=2,
        ...         min_subjects_per_code=2,
        ...         min_occurrences_per_code=None,
        ...         sentinel_code_regex="MEDS_DEATH.*",
        ...         min_events_per_subject=None,
        ...     )
        ...     sorted(pl.read_parquet(out_dir / "data" / "train" / "0.parquet")["code"].to_list())
        ['Lab//A//value_[-inf,3.0)', 'Lab//A//value_[3.0,inf)', 'Lab//A//value_[3.0,inf)']
    """
    with ExitStack() as stack:
        code_filtered_dir = stack.enter_context(tempfile.TemporaryDirectory())
        filter_rare_codes(
            meds_data_dir,
            aggregate_code_metadata_from_meds(meds_data_dir),
            min_subjects_per_code=min_subjects_per_code,
            min_occurrences_per_code=min_occurrences_per_code,
            sentinel_code_regex=sentinel_code_regex,
            output_dir=code_filtered_dir,
        )

        if n_quantile_bins is not None:
            quantized_dir = stack.enter_context(tempfile.TemporaryDirectory())
            quantize_numeric_values(
                code_filtered_dir,
                fit_quantile_metadata(code_filtered_dir, n_bins=n_quantile_bins),
                output_dir=quantized_dir,
            )
            source = quantized_dir
        else:
            source = code_filtered_dir

        return filter_sparse_subjects(
            source,
            min_events_per_subject=min_events_per_subject,
            output_dir=output_dir,
        )
