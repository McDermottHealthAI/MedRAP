"""Subsample a task-labels directory's train split to control the dataset-size axis.

Only ``train.parquet`` is subsampled; other splits (``tuning.parquet``, ``held_out.parquet``)
are copied through unchanged, since the scaling-law fit reuses the same evaluation
resample across models (see the ``test_set_id`` convention in ``simple-scaling-laws``) --
only the amount of training data should vary between dataset-size grid points.
"""

import shutil
from pathlib import Path

import polars as pl


def subsample_task_labels(df: pl.DataFrame, fraction: float, seed: int) -> pl.DataFrame:
    """Return a seeded, deterministic ``fraction`` of ``df``'s rows.

    Examples:
        >>> df = pl.DataFrame({"subject_id": list(range(100)), "task_0": [True, False] * 50})
        >>> len(subsample_task_labels(df, 0.1, seed=0))
        10
        >>> subsample_task_labels(df, 0.1, seed=0).equals(subsample_task_labels(df, 0.1, seed=0))
        True
        >>> subsample_task_labels(df, 1.0, seed=0).equals(df)
        True

        ``fraction`` must be in ``(0, 1]``:

        >>> subsample_task_labels(df, 0.0, seed=0)
        Traceback (most recent call last):
            ...
        ValueError: fraction must be in (0, 1], got 0.0
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return df
    return df.sample(fraction=fraction, seed=seed, shuffle=True)


def write_subsampled_task_labels_dir(
    source_dir: str | Path, output_dir: str | Path, fraction: float, seed: int
) -> None:
    """Write a new task-labels dir with ``train.parquet`` subsampled, other splits copied as-is.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     src = Path(tmp) / "src"
        ...     src.mkdir()
        ...     pl.DataFrame({"subject_id": list(range(100))}).write_parquet(src / "train.parquet")
        ...     pl.DataFrame({"subject_id": list(range(20))}).write_parquet(src / "tuning.parquet")
        ...     out = Path(tmp) / "out"
        ...     write_subsampled_task_labels_dir(src, out, fraction=0.2, seed=0)
        ...     print(len(pl.read_parquet(out / "train.parquet")))
        ...     print(len(pl.read_parquet(out / "tuning.parquet")))
        20
        20
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_path in source_dir.glob("*.parquet"):
        if split_path.stem == "train":
            df = pl.read_parquet(split_path)
            subsample_task_labels(df, fraction, seed).write_parquet(output_dir / split_path.name)
        else:
            shutil.copy2(split_path, output_dir / split_path.name)
