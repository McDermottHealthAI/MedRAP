"""Two-stage MEDS preprocessing pipeline: MEDS-transforms then MTD_preprocess.

Stage 1 (``MEDS_transform-pipeline``): adds time-derived features, filters rare
codes and sparse subjects, quantizes numeric values. Configured via
``conf/_meds_transform_pipeline.yaml``.

Stage 2 (``MTD_preprocess``): tokenizes and tensorizes the intermediate dataset
into ``.nrt`` binary files consumed by meds_torchdata at training time.
"""

import os
import subprocess
from importlib.resources import files as _pkg_files
from pathlib import Path


def run_meds_pipeline(
    meds_data_dir: str | Path,
    output_dir: str | Path,
    *,
    min_subjects_per_code: int,
    min_events_per_subject: int,
) -> tuple[Path, Path]:
    """Run the two-stage MEDS preprocessing pipeline.

    Stage 1: ``MEDS_transform-pipeline`` — adds time-derived features, filters
    rare codes and sparse subjects, quantizes numeric values. Writes to
    ``output_dir/intermediate/``.

    Stage 2: ``MTD_preprocess`` — tokenizes and tensorizes the intermediate
    dataset. Writes to ``output_dir/tensorized/``.

    Args:
        meds_data_dir: Root of a raw MEDS dataset (``data/{split}/*.parquet``).
        output_dir: Parent output directory; ``intermediate/`` and
            ``tensorized/`` subdirectories are created inside it.
        min_subjects_per_code: Passed as ``MIN_SUBJECTS_PER_CODE`` env var to
            the MEDS-transforms pipeline.
        min_events_per_subject: Passed as ``MIN_EVENTS_PER_SUBJECT`` env var to
            the MEDS-transforms pipeline.

    Returns:
        ``(intermediate_dir, tensorized_dir)`` as ``Path`` objects.
    """
    meds_data_dir = Path(meds_data_dir)
    output_dir = Path(output_dir)

    intermediate_dir = output_dir / "intermediate"
    tensorized_dir = output_dir / "tensorized"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    tensorized_dir.mkdir(parents=True, exist_ok=True)

    pipeline_cfg = _pkg_files("medrap").joinpath("conf/_meds_transform_pipeline.yaml")
    subprocess.run(
        ["MEDS_transform-pipeline", str(pipeline_cfg)],
        env={
            **os.environ,
            "RAW_MEDS_DIR": str(meds_data_dir),
            "MTD_INPUT_DIR": str(intermediate_dir),
            "MIN_SUBJECTS_PER_CODE": str(min_subjects_per_code),
            "MIN_EVENTS_PER_SUBJECT": str(min_events_per_subject),
        },
        check=True,
    )

    subprocess.run(
        [
            "MTD_preprocess",
            f"MEDS_dataset_dir={intermediate_dir}",
            f"output_dir={tensorized_dir}",
        ],
        check=True,
    )

    return intermediate_dir, tensorized_dir
