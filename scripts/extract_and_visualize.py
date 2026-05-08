"""Extract retrieval artifacts from a trained MedRAP run and generate diagnostic plots.

Loads the saved config and checkpoint from a training run directory, runs extraction
via ``extract_artifacts()``, and writes a focused set of paper-ready artifacts:

- ``extraction_artifacts.pt`` — the cached prediction tensors.
- ``query_embeddings_{pca,tsne,umap}.pdf`` — single-panel scatter of patient
  query embeddings colored by binary label.
- ``performance.pdf`` — accuracy + AUROC bars for the val split.
- ``top_retrieved_docs.csv`` — top-100 most-retrieved corpus docs ranked by
  top-1 frequency.
- ``retrieval_counts.csv`` — per-position unique-doc count + cumulative.

The script is idempotent: extraction artifacts are cached on disk, so re-runs
finish in ~2 minutes on a CPU node.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import TYPE_CHECKING

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


def plot_query_embeddings(
    query_emb: np.ndarray,
    targets: np.ndarray,
    *,
    method: str,
    output_path: Path,
) -> None:
    """Save a 2-D scatter of query embeddings colored by binary label.

    Args:
        query_emb: ``(N, D_ret)`` patient query embeddings.
        targets: ``(N,)`` binary labels (>0.5 = positive).
        method: ``"pca"``, ``"tsne"``, or ``"umap"``.
        output_path: Destination PDF path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos_mask = targets > 0.5
    neg_mask = ~pos_mask

    method_label = method.upper() if method in {"pca", "tsne", "umap"} else method
    proj = _reduce_2d(query_emb, method)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        proj[neg_mask, 0],
        proj[neg_mask, 1],
        c="red",
        marker="o",
        alpha=0.3,
        s=40,
        edgecolors="none",
        label="Label 0",
    )
    ax.scatter(
        proj[pos_mask, 0],
        proj[pos_mask, 1],
        c="blue",
        marker="o",
        alpha=0.3,
        s=40,
        edgecolors="none",
        label="Label 1",
    )
    ax.set_xlabel(f"{method_label}-1")
    ax.set_ylabel(f"{method_label}-2")
    ax.set_title(f"Query Embeddings ({method_label})")
    ax.legend(fontsize=8)

    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Query embedding plot saved to {output_path}")


def plot_performance(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    output_path: Path,
) -> None:
    """Save a single-panel accuracy + AUROC bar chart.

    Args:
        logits: ``(N, C)`` model output. ``C`` may be 1 (binary logit) or 2
            (logit per class).
        targets: ``(N,)`` binary labels (>0.5 = positive).
        output_path: Destination PDF path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
        # Sort by descending predicted probability.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and visualize MedRAP retrieval artifacts.")
    parser.add_argument("--run_dir", type=str, required=True, help="Training run output directory.")
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

    method_outputs = {
        "pca": "query_embeddings_pca.pdf",
        "tsne": "query_embeddings_tsne.pdf",
        "umap": "query_embeddings_umap.pdf",
    }
    for method, fname in method_outputs.items():
        try:
            plot_query_embeddings(
                query_emb,
                targets,
                method=method,
                output_path=extract_dir / fname,
            )
        except ImportError as exc:
            print(
                f"Skipping {method} plot ({fname}): {exc}. "
                f"Install the missing package to enable it.",
                file=sys.stderr,
            )

    plot_performance(logits, targets, output_path=extract_dir / "performance.pdf")

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
