"""Run the full extraction pipeline for a trained MedRAP run.

Wraps :mod:`scripts.extract_and_visualize` and :mod:`scripts.run_demographic_heatmap`
into a single command. Reads ``--run_dir``, automatically discovers the checkpoint
and (when possible) retrieval DB paths from ``<run_dir>/config.yaml``, and produces
all 9 files in ``<run_dir>/extraction/``:

- ``extraction_artifacts.pt`` (with cached ``per_doc_loo_delta_logits``)
- ``extraction_diagnostics.png`` (PCA), ``_tsne.png``, ``_umap.png``
- ``top_retrieved_docs.csv``
- ``keyword_demographic_heatmap.png``
- ``keyword_demographic_{age,gender,race}.csv``

Both wrapped scripts already cache their work, so re-runs hit the cache and
finish in ~2 minutes on a CPU node.

Usage::

    python scripts/run_extraction_pipeline.py \\
        --run_dir outputs/mimic_run_rope_cross_attention \\
        --meds_cohort /groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

_SCRIPTS_DIR = Path(__file__).resolve().parent


def _resolve_retrieval_db(run_dir: Path, override: Path | None) -> Path:
    """Return the retrieval-DB path, preferring CLI override over config lookup.

    Args:
        run_dir: Training run directory containing ``config.yaml``.
        override: Optional path passed via ``--retrieval_db``.

    Returns:
        Resolved path. Raises ``ValueError`` if neither override is given nor
        ``retriever.dataset_path`` is present in the config.
    """
    if override is not None:
        return override
    cfg = OmegaConf.load(run_dir / "config.yaml")
    db = OmegaConf.select(cfg, "retriever.dataset_path")
    if db is None:
        raise ValueError(
            f"could not auto-detect retrieval DB: {run_dir / 'config.yaml'} has no "
            "`retriever.dataset_path`. Pass --retrieval_db explicitly."
        )
    return Path(db)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate all extraction-directory artifacts for a trained MedRAP run."
    )
    parser.add_argument(
        "--run_dir", type=Path, required=True, help="Training run directory (must contain config.yaml)."
    )
    parser.add_argument(
        "--meds_cohort",
        type=Path,
        required=True,
        help="Raw MEDS cohort directory used by the demographic heatmap step.",
    )
    parser.add_argument(
        "--retrieval_db",
        type=Path,
        default=None,
        help="HF retrieval DB. Defaults to <run_dir>/config.yaml `retriever.dataset_path`.",
    )
    parser.add_argument(
        "--keyword_provider",
        type=str,
        default="lda",
        help="Doc → keyword provider for the heatmap: 'lda' (default) or 'title'.",
    )
    parser.add_argument(
        "--n_topics",
        type=int,
        default=30,
        help="Number of LDA topics (only used with --keyword_provider lda).",
    )
    parser.add_argument(
        "--top_n_keywords",
        type=int,
        default=20,
        help="Cap on number of keywords shown per heatmap.",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        print(f"Error: {config_path} not found.", file=sys.stderr)
        return 1

    retrieval_db = _resolve_retrieval_db(run_dir, args.retrieval_db)

    extract_cmd = [
        sys.executable,
        str(_SCRIPTS_DIR / "extract_and_visualize.py"),
        "--run_dir",
        str(run_dir),
    ]
    print(">>> step 1/2:", " ".join(extract_cmd))
    subprocess.run(extract_cmd, check=True)

    heatmap_cmd = [
        sys.executable,
        str(_SCRIPTS_DIR / "run_demographic_heatmap.py"),
        "--run_dir",
        str(run_dir),
        "--retrieval_db",
        str(retrieval_db),
        "--meds_cohort",
        str(args.meds_cohort),
        "--keyword_provider",
        args.keyword_provider,
        "--n_topics",
        str(args.n_topics),
        "--top_n_keywords",
        str(args.top_n_keywords),
    ]
    print(">>> step 2/2:", " ".join(heatmap_cmd))
    subprocess.run(heatmap_cmd, check=True)

    extract_dir = run_dir / "extraction"
    print(f"Done. Wrote artifacts to {extract_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
