"""Extract retrieval artifacts from a trained MedRAP run and generate diagnostic plots.

Loads the saved config and checkpoint from a training run directory, runs extraction
via ``extract_artifacts()``, and writes a focused set of paper-ready artifacts.

Binary single-task runs produce:

- ``extraction_artifacts.pt`` — the cached prediction tensors.
- ``query_embeddings_{pca,tsne,umap}.pdf`` — single-panel scatter colored by
  the binary label.
- ``performance.pdf`` — accuracy + AUROC bars.
- ``top_retrieved_docs.csv``, ``retrieval_counts.csv``.

Multitask runs (``targets`` shape ``(N, T)``, NaN-masked) produce the same
``extraction_artifacts.pt`` plus:

- ``query_embeddings_task{K}_{pca,tsne,umap}.pdf`` for each task ``K`` —
  scatter colored by whether task ``K`` is positive (NaN treated as
  not-positive). The 2-D projection is computed once per method and re-used
  across tasks.
- ``performance.pdf`` — per-task AUROC bar chart with a horizontal red
  dashed line at the mean over tasks where both classes are present.
- Same ``top_retrieved_docs.csv`` / ``retrieval_counts.csv``.

Task type is auto-detected from ``artifacts['targets'].ndim`` and can be
overridden with ``--task_mode {binary, multitask}``.

The script is idempotent: extraction artifacts are cached on disk, so re-runs
finish in ~2 minutes on a CPU node.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Default location of MIMIC-IV's d_labitems on this cluster. Used to translate
# `LAB//<itemid>//<unit>` MEDS codes into readable test names (e.g.
# "Glucose [Urine]"). Override with `--mimic_labitems_path`.
_DEFAULT_LABITEMS_PATH = Path(
    "/groups/mm6677_gp/data/MIMIC_MEDS/raw_input/hosp/d_labitems.csv.gz"
)

import lightning
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from omegaconf import OmegaConf
from torch import Tensor

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

# Ensure the project is importable when run from the repo root.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from medrap.configs import instantiate_datamodule, instantiate_training_module  # noqa: E402
from medrap.demographic_analysis import LDATopicProvider  # noqa: E402
from medrap.extraction import extract_artifacts  # noqa: E402

# ---------------------------------------------------------------------------
# Checkpoint resolution (mirrors cli._find_checkpoint_path)
# ---------------------------------------------------------------------------


def _find_checkpoint(run_dir: Path) -> Path:
    """Return the best available checkpoint in *run_dir*."""
    best = run_dir / "best_model.ckpt"
    if best.is_file():
        return best
    last = run_dir / "checkpoints" / "last.ckpt"
    if last.is_file():
        return last
    epoch_ckpts = sorted((run_dir / "checkpoints").glob("epoch=*-step=*.ckpt"))
    if epoch_ckpts:
        return epoch_ckpts[-1]
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def run_extraction(run_dir: Path) -> tuple[dict[str, Tensor], Path, Path | None]:
    """Load a trained model and extract artifacts from the val split.

    Returns ``(artifacts, artifact_path, retrieval_db_path)``.
    ``retrieval_db_path`` is ``None`` for runs that use an ``InMemoryRetriever``.
    """
    cfg = OmegaConf.load(run_dir / "config.yaml")
    ckpt_path = _find_checkpoint(run_dir)

    module = instantiate_training_module(cfg)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    module.load_state_dict(checkpoint["state_dict"])

    retrieval_db_path: Path | None = None
    dataset_path = OmegaConf.select(cfg, "retriever.dataset_path")
    if dataset_path is not None:
        retrieval_db_path = Path(dataset_path)

    # Datamodule — val split (deterministic, no shuffle).
    datamodule = instantiate_datamodule(cfg)
    datamodule.setup("fit")
    dataloader: DataLoader = datamodule.val_dataloader()

    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    extract_dir = run_dir / "extraction"
    cache_existed = (extract_dir / "extraction_artifacts.pt").is_file()
    artifact_path = extract_artifacts(
        module, dataloader, trainer, output_dir=extract_dir, use_cache=True
    )
    if cache_existed:
        print(f"Using cached artifacts at {artifact_path}; skipped trainer.predict.")
    artifacts = torch.load(artifact_path, weights_only=True)

    return artifacts, artifact_path, retrieval_db_path


# ---------------------------------------------------------------------------
# Task-mode resolution
# ---------------------------------------------------------------------------


def _resolve_task_mode(mode: str, targets: np.ndarray) -> str:
    """Map ``--task_mode auto|binary|multitask`` to ``binary`` or ``multitask``.

    For ``auto``: 2-D ``targets`` (``(N, T)``) → ``multitask``; anything else
    → ``binary``.
    """
    if mode == "auto":
        return "multitask" if targets.ndim == 2 else "binary"
    if mode in {"binary", "multitask"}:
        return mode
    raise ValueError(f"unknown task_mode {mode!r}; expected one of auto/binary/multitask")


def _load_task_codes(run_dir: Path) -> dict[int, str] | None:
    """Return ``{task_idx: meds_code}`` from the multitask labels dir, or ``None``.

    The mapping is written by ``scripts/prepare_multi_task_labels.py`` as
    ``code_index.json`` next to the task-label parquets. Its location is
    recorded in the run config at ``training.datamodule.mt_labels_dir``.
    Returns ``None`` for binary runs (which don't set that field) or if the
    file is missing / malformed.
    """
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        return None
    cfg = OmegaConf.load(config_path)
    mt_labels_dir = OmegaConf.select(cfg, "training.datamodule.mt_labels_dir")
    if mt_labels_dir is None:
        return None
    code_index_path = Path(mt_labels_dir) / "code_index.json"
    if not code_index_path.is_file():
        return None
    try:
        raw = json.loads(code_index_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return {int(k): str(v) for k, v in raw.items()}


def _load_lab_label_lookup(path: Path) -> dict[int, tuple[str, str]] | None:
    """Read MIMIC-IV's ``d_labitems`` into ``{itemid: (label, fluid)}``.

    Accepts either a plain CSV or a gzip-compressed CSV (``.csv.gz``).
    Returns ``None`` when the file is missing or unreadable. Translates
    ``LAB//<itemid>//<unit>`` MEDS codes into human-readable test names like
    ``Glucose [Urine]``.
    """
    if not path.is_file():
        return None
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", newline="") as f:
            reader = csv.DictReader(f)
            lookup: dict[int, tuple[str, str]] = {}
            for row in reader:
                try:
                    itemid = int(row["itemid"])
                except (KeyError, ValueError):
                    continue
                label = (row.get("label") or "").strip()
                fluid = (row.get("fluid") or "").strip()
                if label:
                    lookup[itemid] = (label, fluid)
        return lookup or None
    except (OSError, csv.Error):
        return None


def _humanize_meds_code(
    code: str, lab_lookup: dict[int, tuple[str, str]] | None = None
) -> str:
    """Convert a MEDS code into a readable label for figure titles / ticks.

    Examples:
        - ``LAB//51478//mg/dL`` → ``"Glucose [Urine]"`` (when ``lab_lookup``
          is provided), else ``"LAB 51478"``.
        - ``MEDICATION//START//Heparin`` → ``"Heparin (start)"``.
        - ``MEDICATION//<drug>//Administered`` → ``"<drug> (administered)"``.
        - ``TRANSFER_TO//discharge//UNKNOWN`` → ``"Transfer to discharge"``.
        - ``ED_OUT`` → ``"ED departure"``.
        - Anything unrecognized falls back to the raw code.
    """
    parts = code.split("//")
    head = parts[0]

    if head == "LAB" and len(parts) >= 2:
        try:
            itemid = int(parts[1])
        except ValueError:
            return code
        if lab_lookup and itemid in lab_lookup:
            label, fluid = lab_lookup[itemid]
            return f"{label} [{fluid}]" if fluid else label
        return f"LAB {itemid}"

    if head == "MEDICATION" and len(parts) >= 3:
        # Three observed shapes:
        #   MEDICATION//START//<drug>
        #   MEDICATION//STOP//<drug>
        #   MEDICATION//<drug>//Administered
        action_or_drug = parts[1].strip()
        third = parts[2].strip()
        if action_or_drug in ("START", "STOP"):
            drug = third or "unknown"
            verb = action_or_drug.lower()
            if drug == "UNK":
                return f"Medication {verb} (UNK)"
            return f"{drug} ({verb})"
        if third == "Administered":
            return f"{action_or_drug} (administered)"

    if head == "TRANSFER_TO" and len(parts) >= 2:
        target = parts[1].strip().lower() or "unknown"
        return f"Transfer to {target}"

    if head == "ED_OUT":
        return "ED departure"

    return code


def _build_task_labels(
    run_dir: Path,
    lab_lookup: dict[int, tuple[str, str]] | None,
) -> dict[int, str] | None:
    """Compose ``code_index.json`` with a ``d_labitems`` lookup into readable labels.

    Returns ``{task_idx: readable_label}`` or ``None`` for binary runs / when
    ``code_index.json`` is missing.
    """
    codes = _load_task_codes(run_dir)
    if codes is None:
        return None
    return {idx: _humanize_meds_code(code, lab_lookup) for idx, code in codes.items()}


def _short_task_label(task_idx: int, task_names: dict[int, str] | None, *, max_chars: int = 30) -> str:
    """Return a compact ``"<idx>: <name>"`` label, truncated for tick text."""
    if not task_names or task_idx not in task_names:
        return str(task_idx)
    name = task_names[task_idx]
    if len(name) > max_chars:
        name = name[: max_chars - 3] + "..."
    return f"{task_idx}: {name}"


# ---------------------------------------------------------------------------
# 2-D dimensionality reduction (PCA / t-SNE / UMAP)
# ---------------------------------------------------------------------------


def _pca_2d(x: np.ndarray) -> np.ndarray:
    """Project rows of *x* to 2D via PCA (mean-centered SVD).  No sklearn dep."""
    x_centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    return x_centered @ vt[:2].T


def _tsne_2d(x: np.ndarray) -> np.ndarray:
    """Project rows of *x* to 2D via t-SNE (sklearn)."""
    from sklearn.manifold import TSNE

    perplexity = float(min(30, max(5, (x.shape[0] - 1) / 3)))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=0,
    ).fit_transform(x)


def _umap_2d(x: np.ndarray) -> np.ndarray:
    """Project rows of *x* to 2D via UMAP (umap-learn)."""
    import umap

    n_neighbors = int(min(15, max(2, x.shape[0] - 1)))
    return umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=0,
    ).fit_transform(x)


_REDUCERS = {"pca": _pca_2d, "tsne": _tsne_2d, "umap": _umap_2d}


def _reduce_2d(x: np.ndarray, method: str) -> np.ndarray:
    if method not in _REDUCERS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(_REDUCERS)}")
    return _REDUCERS[method](x)


# ---------------------------------------------------------------------------
# Single-panel plotters
# ---------------------------------------------------------------------------


def _scatter_query_embeddings(
    proj: np.ndarray,
    pos_mask: np.ndarray,
    *,
    method: str,
    output_path: Path,
    title_suffix: str = "",
    pos_label: str = "Label 1",
    neg_label: str = "Label 0",
) -> None:
    """Save a 2-D scatter colored by a 1-D bool ``pos_mask``.

    Args:
        proj: ``(N, 2)`` precomputed 2-D projection of the query embeddings.
        pos_mask: ``(N,)`` bool mask — True = positive, False = negative.
        method: reducer name used in axis labels and the title (``"pca"``,
            ``"tsne"``, ``"umap"``).
        output_path: Destination PDF path.
        title_suffix: Optional extra text appended to the figure title (e.g.
            ``"task 7"``). Empty string for binary plots.
        pos_label: Legend label for ``pos_mask`` True points.
        neg_label: Legend label for ``pos_mask`` False points.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    neg_mask = ~pos_mask
    method_label = method.upper() if method in {"pca", "tsne", "umap"} else method

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        proj[neg_mask, 0],
        proj[neg_mask, 1],
        c="red",
        marker="o",
        alpha=0.3,
        s=40,
        edgecolors="none",
        label=neg_label,
    )
    ax.scatter(
        proj[pos_mask, 0],
        proj[pos_mask, 1],
        c="blue",
        marker="o",
        alpha=0.3,
        s=40,
        edgecolors="none",
        label=pos_label,
    )
    ax.set_xlabel(f"{method_label}-1")
    ax.set_ylabel(f"{method_label}-2")
    title = f"Query Embeddings ({method_label})"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)
    ax.legend(fontsize=8)

    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Query embedding plot saved to {output_path}")


def plot_performance(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    task_mode: str,
    output_path: Path,
    task_names: dict[int, str] | None = None,
) -> None:
    """Save a performance summary PDF.

    For ``task_mode == "binary"``: an accuracy + AUROC bar chart.

    For ``task_mode == "multitask"``: a per-task AUROC bar chart (task index
    on the x-axis), with a red dashed horizontal line at the mean AUROC over
    tasks where both classes are present. Per-task AUROC is computed by
    :func:`medrap.metrics.multitask_auroc_torch`, which already NaN-masks
    missing labels.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if task_mode == "multitask":
        from medrap.metrics import multitask_auroc_torch

        n_tasks = targets.shape[1]
        auroc_by_task = multitask_auroc_torch(
            torch.from_numpy(targets),
            torch.from_numpy(logits),
        )
        per_task = np.full(n_tasks, np.nan)
        for task_idx, value in auroc_by_task.items():
            per_task[task_idx] = float(value)

        present = ~np.isnan(per_task)
        mean_auroc = float(np.nanmean(per_task)) if present.any() else float("nan")

        fig_width = max(8.0, n_tasks * 0.5)
        fig_height = 6.0 if task_names else 4.0
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        x = np.arange(n_tasks)
        ax.bar(x, np.nan_to_num(per_task, nan=0.0), color="steelblue")
        if present.any():
            ax.axhline(
                mean_auroc,
                color="red",
                linestyle="--",
                label=f"Mean AUROC = {mean_auroc:.3f} ({int(present.sum())}/{n_tasks} tasks)",
            )
            ax.legend(fontsize=8)
        ax.set_xticks(x)
        tick_labels = [_short_task_label(i, task_names) for i in range(n_tasks)]
        if task_names:
            ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")
        else:
            ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_xlabel("Task")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0, 1.05)
        ax.set_title("Per-task AUROC")

        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(
            f"Performance plot saved to {output_path} "
            f"(multitask: {int(present.sum())}/{n_tasks} tasks with both classes; "
            f"mean AUROC = {mean_auroc:.3f})"
        )
        return

    # Binary path.
    if logits.shape[1] == 2:
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        pred_class = logits.argmax(axis=1)
        pos_prob = probs[:, 1]
    elif logits.shape[1] == 1:
        pos_prob = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        pred_class = (pos_prob >= 0.5).astype(int)
    else:
        pos_prob = None
        pred_class = logits.argmax(axis=1)

    true_class = (targets > 0.5).astype(int)
    accuracy = float((pred_class == true_class).mean())

    auroc = None
    if pos_prob is not None and len(np.unique(true_class)) == 2:
        order = np.argsort(-pos_prob)
        sorted_labels = true_class[order]
        n_pos_total = sorted_labels.sum()
        n_neg_total = len(sorted_labels) - n_pos_total
        if n_pos_total > 0 and n_neg_total > 0:
            tp_cumsum = np.cumsum(sorted_labels)
            fp_cumsum = np.cumsum(1 - sorted_labels)
            tpr = tp_cumsum / n_pos_total
            fpr = fp_cumsum / n_neg_total
            tpr = np.concatenate([[0.0], tpr])
            fpr = np.concatenate([[0.0], fpr])
            auroc = float(np.trapezoid(tpr, fpr))

    metrics = {"Accuracy": accuracy}
    if auroc is not None:
        metrics["AUROC"] = auroc

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(range(len(metrics)), list(metrics.values()), color=["steelblue", "coral"][: len(metrics)])
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(list(metrics.keys()))
    ax.set_ylim(0, 1.05)
    ax.set_title("Prediction Summary")
    for bar, val in zip(bars, metrics.values(), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Performance plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Retrieval coverage CSV
# ---------------------------------------------------------------------------


def write_retrieval_counts(doc_ids: Tensor | np.ndarray, output_path: Path) -> None:
    """Write per-position unique-doc counts (current and cumulative) as CSV.

    The output mirrors a stdout summary: a banner header
    ``# N=<N> patients, K=<K>`` followed by a 3-column table whose row ``p``
    reports

    - ``unique_at_pos``: ``len(unique(doc_ids[:, p-1]))``
    - ``unique_cumulative``: ``len(unique(doc_ids[:, :p].flatten()))``

    Args:
        doc_ids: Tensor or array of shape ``(N, K)`` or ``(N, 1, K)`` (the
            ``R`` axis is squeezed when present).
        output_path: Destination CSV path.
    """
    arr = doc_ids.numpy() if isinstance(doc_ids, Tensor) else np.asarray(doc_ids)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    elif arr.ndim != 2:
        raise ValueError(f"doc_ids must be (N, K) or (N, 1, K); got shape {tuple(arr.shape)}")
    n, k = arr.shape

    with open(output_path, "w", newline="") as f:
        f.write(f"# N={n} patients, K={k}\n")
        writer = csv.writer(f)
        writer.writerow(["pos", "unique_at_pos", "unique_cumulative"])
        for p in range(1, k + 1):
            unique_at_pos = int(np.unique(arr[:, p - 1]).size)
            unique_cumulative = int(np.unique(arr[:, :p].reshape(-1)).size)
            writer.writerow([p, unique_at_pos, unique_cumulative])
    print(f"Retrieval counts CSV saved to {output_path}")


# ---------------------------------------------------------------------------
# Top retrieved docs CSV export
# ---------------------------------------------------------------------------


def write_top_retrieved_docs(
    artifacts: dict[str, Tensor],
    output_path: Path,
    *,
    retrieval_db_path: Path,
    n_top: int = 100,
    n_topics: int = 30,
) -> Path:
    """Write a CSV of the top-``n_top`` most-retrieved docs, ranked by top-1 frequency.

    Each row carries the textbook title, retrieval counts (top-1 and top-K),
    LDA topic keywords, and the raw ``content`` of the doc.

    Rows with zero top-K retrievals are dropped so a collapsed retriever does
    not pad the CSV with empty rows.
    """
    doc_ids_tensor = artifacts["doc_ids"]  # (N, R, K)
    if doc_ids_tensor.ndim != 3 or doc_ids_tensor.shape[1] != 1:
        raise ValueError(
            f"Expected doc_ids with shape (N, 1, K); got {tuple(doc_ids_tensor.shape)}."
        )
    doc_ids = doc_ids_tensor[:, 0, :].cpu().numpy().astype(np.int64)  # (N, K)
    n_patients, k_docs = doc_ids.shape

    ds = load_from_disk(str(retrieval_db_path))
    corpus_size = len(ds)

    top_1_counts = np.bincount(doc_ids[:, 0], minlength=corpus_size)
    top_k_counts = np.bincount(doc_ids.reshape(-1), minlength=corpus_size)

    order = np.lexsort([-top_k_counts, -top_1_counts])
    n_nonzero = int((top_k_counts > 0).sum())
    n_rows = min(n_top, n_nonzero)
    order = order[:n_rows]

    provider = LDATopicProvider(retrieval_db_path, n_topics=n_topics)

    rows = []
    for rank, doc_id in enumerate(order, start=1):
        doc_id = int(doc_id)
        entry = ds[doc_id]
        keyword_pairs = provider.keywords_for(doc_id)
        keywords_str = "; ".join(f"{label} ({weight:.2f})" for label, weight in keyword_pairs)
        rows.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "title": entry.get("title", ""),
                "top_1_count": int(top_1_counts[doc_id]),
                "top_k_count": int(top_k_counts[doc_id]),
                "top_1_rate": float(top_1_counts[doc_id]) / n_patients,
                "top_k_rate": float(top_k_counts[doc_id]) / (n_patients * k_docs),
                "lda_keywords": keywords_str,
                "content": entry.get("content", entry.get("contents", "")),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "rank",
            "doc_id",
            "title",
            "top_1_count",
            "top_k_count",
            "top_1_rate",
            "top_k_rate",
            "lda_keywords",
            "content",
        ],
    )
    df.to_csv(output_path, index=False)
    print(f"Top-{n_rows} retrieved docs CSV saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_METHODS: tuple[str, ...] = ("pca", "tsne", "umap")


def _render_query_embedding_plots(
    query_emb: np.ndarray,
    targets: np.ndarray,
    *,
    task_mode: str,
    extract_dir: Path,
    task_names: dict[int, str] | None = None,
) -> None:
    """Render query-embedding scatter PDFs for all methods, looped over tasks for multitask.

    The 2-D projection is computed once per method and reused across tasks so
    t-SNE / UMAP cost is paid 3× total, not 75× for a 25-task run.
    """
    projections: dict[str, np.ndarray] = {}
    for method in _METHODS:
        try:
            projections[method] = _reduce_2d(query_emb, method)
        except ImportError as exc:
            print(
                f"Skipping {method} projection: {exc}. "
                f"Install the missing package to enable it.",
                file=sys.stderr,
            )

    if task_mode == "binary":
        pos_mask = targets > 0.5
        for method, proj in projections.items():
            _scatter_query_embeddings(
                proj,
                pos_mask,
                method=method,
                output_path=extract_dir / f"query_embeddings_{method}.pdf",
            )
        return

    # Multitask: one PDF per (task, method). Patients whose label for the task is
    # NaN (label not extracted for them) are dropped from the scatter entirely so
    # they don't contaminate either color class.
    n_tasks = targets.shape[1]
    for task_idx in range(n_tasks):
        col = targets[:, task_idx]
        valid = ~np.isnan(col)
        n_valid = int(valid.sum())
        n_nan = int(col.shape[0] - n_valid)
        if n_valid == 0:
            print(
                f"Skipping task {task_idx}: all {col.shape[0]} labels are NaN.",
                file=sys.stderr,
            )
            continue
        valid_col = col[valid]
        pos_mask = valid_col > 0.5
        n_pos = int(pos_mask.sum())
        task_name = task_names.get(task_idx) if task_names else None
        descriptor = f"task {task_idx}" if task_name is None else f"task {task_idx} ({task_name})"
        print(
            f"{descriptor}: plotting n={n_valid} (pos={n_pos}, neg={n_valid - n_pos}); "
            f"excluded {n_nan} NaN-labeled patients."
        )
        for method, proj in projections.items():
            _scatter_query_embeddings(
                proj[valid],
                pos_mask,
                method=method,
                output_path=extract_dir / f"query_embeddings_task{task_idx}_{method}.pdf",
                title_suffix=f"{descriptor} (n={n_valid})",
                pos_label=f"{descriptor} positive",
                neg_label=f"{descriptor} negative",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and visualize MedRAP retrieval artifacts.")
    parser.add_argument("--run_dir", type=str, required=True, help="Training run output directory.")
    parser.add_argument(
        "--task_mode",
        choices=("auto", "binary", "multitask"),
        default="auto",
        help=(
            "Plot rendering mode. Default 'auto' inspects artifacts['targets'].ndim — 1-D → "
            "binary; 2-D → multitask (per-task query-embedding PDFs + per-task AUROC chart). "
            "Use 'binary' / 'multitask' to override."
        ),
    )
    parser.add_argument(
        "--mimic_labitems_path",
        type=Path,
        default=_DEFAULT_LABITEMS_PATH,
        help=(
            "Path to MIMIC-IV's d_labitems CSV (gzip OK). Used to translate "
            "LAB//<itemid>//<unit> task codes into readable test names. "
            f"Default: {_DEFAULT_LABITEMS_PATH}. Pass a non-existent path to "
            "skip the lookup; tasks will then fall back to 'LAB <itemid>'."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "config.yaml").is_file():
        print(f"Error: {run_dir / 'config.yaml'} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Run directory: {run_dir}")
    artifacts, artifact_path, retrieval_db_path = run_extraction(run_dir)
    print(f"Artifacts saved to {artifact_path}")
    print(f"Keys: {sorted(artifacts.keys())}")
    for key, value in sorted(artifacts.items()):
        print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")

    extract_dir = artifact_path.parent
    targets = artifacts["targets"].numpy()
    logits = artifacts["logits"].numpy()
    query_emb = artifacts["query_embeddings"].numpy()
    if query_emb.ndim == 3 and query_emb.shape[1] == 1:
        query_emb = query_emb[:, 0, :]  # (N, D_ret)

    task_mode = _resolve_task_mode(args.task_mode, targets)
    print(f"Task mode: {task_mode} (requested: {args.task_mode})")

    task_names: dict[int, str] | None = None
    if task_mode == "multitask":
        lab_lookup = _load_lab_label_lookup(args.mimic_labitems_path)
        if lab_lookup is None:
            print(
                f"Note: d_labitems lookup not loaded from {args.mimic_labitems_path}. "
                "LAB tasks will fall back to 'LAB <itemid>'.",
                file=sys.stderr,
            )
        else:
            print(f"Loaded d_labitems lookup ({len(lab_lookup)} entries) from {args.mimic_labitems_path}.")
        task_names = _build_task_labels(run_dir, lab_lookup)
        if task_names is None:
            print(
                "Note: no task-name mapping found at <run_dir>/config.yaml :: "
                "training.datamodule.mt_labels_dir/code_index.json. "
                "Tasks will be labeled by index only.",
                file=sys.stderr,
            )
        else:
            print(f"Loaded task-name mapping for {len(task_names)} tasks.")

    _render_query_embedding_plots(
        query_emb,
        targets,
        task_mode=task_mode,
        extract_dir=extract_dir,
        task_names=task_names,
    )

    plot_performance(
        logits,
        targets,
        task_mode=task_mode,
        output_path=extract_dir / "performance.pdf",
        task_names=task_names,
    )

    if retrieval_db_path is not None:
        write_top_retrieved_docs(
            artifacts,
            extract_dir / "top_retrieved_docs.csv",
            retrieval_db_path=retrieval_db_path,
            n_top=100,
        )
    else:
        print("Skipping top_retrieved_docs.csv: no retrieval DB path (InMemoryRetriever run).")

    write_retrieval_counts(artifacts["doc_ids"], extract_dir / "retrieval_counts.csv")


if __name__ == "__main__":
    main()
