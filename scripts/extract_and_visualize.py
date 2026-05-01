"""Extract retrieval artifacts from a trained MedRAP run and generate diagnostic plots.

Loads the saved config and checkpoint from a training run directory, runs extraction
via ``extract_artifacts()``, and produces a 6-panel diagnostic figure.

Usage::

    python scripts/extract_and_visualize.py --run_dir outputs/margin_check_XXX

The script also works on real-data runs (e.g. MIMIC retrieval-only).

Cross-attention runs (which fuse all retrieved docs jointly) do not produce a
native ``per_doc_logits``. For those checkpoints the script computes a
leave-one-out (LOO) Δlogit proxy via :func:`medrap.extraction.persist_loo_per_doc_logits`,
caches it back into the saved ``.pt`` artifact under ``per_doc_loo_delta_logits``,
and aliases it to ``per_doc_logits`` in memory so the existing per-doc plot
panels render. Each value answers: "how much does this doc shift the patient's
positive logit, relative to a forward pass without it?" Positive Δ = doc pushes
the patient toward the positive class.
"""

from __future__ import annotations

import argparse
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
from medrap.extraction import extract_artifacts, persist_loo_per_doc_logits  # noqa: E402

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


def run_extraction(
    run_dir: Path,
) -> tuple[dict[str, Tensor], Path, Tensor | None, Path | None]:
    """Load a trained model and extract artifacts from the val split.

    Returns (artifacts, artifact_path, all_doc_key_embeddings, retrieval_db_path).
    ``retrieval_db_path`` is ``None`` for runs that use an ``InMemoryRetriever``.
    """
    cfg = OmegaConf.load(run_dir / "config.yaml")
    ckpt_path = _find_checkpoint(run_dir)

    # Rebuild model + load weights.
    module = instantiate_training_module(cfg)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    module.load_state_dict(checkpoint["state_dict"])

    # Grab full corpus key embeddings from the retriever (if InMemoryRetriever).
    all_doc_keys: Tensor | None = None
    retriever = getattr(module.model, "retriever", None)
    if retriever is not None and hasattr(retriever, "_doc_key_embeddings"):
        all_doc_keys = retriever._doc_key_embeddings.detach().clone()

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

    if "per_doc_logits" not in artifacts:
        delta = persist_loo_per_doc_logits(artifact_path, module, dataloader)
        artifacts["per_doc_loo_delta_logits"] = delta
        artifacts["per_doc_logits"] = delta
        print(
            "Computed leave-one-out Δlogit per doc via 1 full + 1 vectorized LOO "
            f"forward per batch; cached at {artifact_path} under key "
            "'per_doc_loo_delta_logits'."
        )

    return artifacts, artifact_path, all_doc_keys, retrieval_db_path


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
# Visualization
# ---------------------------------------------------------------------------


def generate_plots(
    artifacts: dict[str, Tensor],
    output_path: Path,
    all_doc_key_embeddings: np.ndarray | None = None,
    *,
    method: str = "pca",
) -> None:
    """Create a 6-panel diagnostic figure and save to *output_path*.

    Args:
        all_doc_key_embeddings: Full corpus key matrix ``(N_docs, D)`` from the
            retriever.  When provided, Plot 2 uses the full corpus rather than
            only the retrieved subset.
        method: 2-D dim reduction for Plots 1 and 2 — ``"pca"``, ``"tsne"``, or
            ``"umap"``.  The other panels are unaffected.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_label = method.upper() if method in {"pca", "tsne", "umap"} else method
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"MedRAP Retrieval Extraction Diagnostics ({method_label})", fontsize=14, y=0.98
    )

    targets = artifacts["targets"].numpy()
    logits = artifacts["logits"].numpy()
    pos_mask = targets > 0.5
    neg_mask = ~pos_mask

    # Squeeze R=1 dim from retrieval tensors if present.
    query_emb = artifacts["query_embeddings"].numpy()
    if query_emb.ndim == 3 and query_emb.shape[1] == 1:
        query_emb = query_emb[:, 0, :]  # (N, D_ret)

    has_doc_ids = "doc_ids" in artifacts
    has_per_doc = "per_doc_logits" in artifacts and "differentiable_doc_scores" in artifacts

    if has_doc_ids:
        doc_ids = artifacts["doc_ids"].numpy()
        if doc_ids.ndim == 3 and doc_ids.shape[1] == 1:
            doc_ids = doc_ids[:, 0, :]  # (N, K)

    doc_key_emb = artifacts.get("doc_key_embeddings")
    if doc_key_emb is not None:
        doc_key_emb = doc_key_emb.numpy()
        if doc_key_emb.ndim == 4 and doc_key_emb.shape[1] == 1:
            doc_key_emb = doc_key_emb[:, 0, :, :]  # (N, K, D_ret)

    if has_per_doc:
        per_doc_logits = artifacts["per_doc_logits"].numpy()  # (N, K, C)
        diff_scores = artifacts["differentiable_doc_scores"].numpy()  # (N, K)

    # ------------------------------------------------------------------
    # Plot 1: Query embedding PCA scatter
    # ------------------------------------------------------------------
    ax1 = axes[0, 0]
    q_proj = _reduce_2d(query_emb, method)
    ax1.scatter(
        q_proj[neg_mask, 0],
        q_proj[neg_mask, 1],
        c="red",
        marker="o",
        alpha=0.3,
        s=40,
        edgecolors="none",
        label="Label 0",
    )
    ax1.scatter(
        q_proj[pos_mask, 0],
        q_proj[pos_mask, 1],
        c="blue",
        marker="o",
        alpha=0.3,
        s=40,
        edgecolors="none",
        label="Label 1",
    )
    ax1.set_xlabel(f"{method_label}-1")
    ax1.set_ylabel(f"{method_label}-2")
    ax1.set_title(f"Query Embeddings ({method_label})")
    ax1.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Plot 2: Doc key embedding PCA scatter, colored by avg LOO Δlogit per doc
    # ------------------------------------------------------------------
    ax2 = axes[0, 1]
    # Use full corpus keys if available, otherwise unique retrieved keys.
    if all_doc_key_embeddings is not None:
        corpus_keys = all_doc_key_embeddings
    elif doc_key_emb is not None:
        corpus_keys = np.unique(doc_key_emb.reshape(-1, doc_key_emb.shape[-1]), axis=0)
    else:
        corpus_keys = None

    if corpus_keys is not None and has_per_doc:
        n_corpus_docs = len(corpus_keys)
        c_idx = min(1, per_doc_logits.shape[-1] - 1)

        # Average positive-class logit per unique doc across all patients that retrieved it.
        doc_avg_logit = np.zeros(n_corpus_docs)
        doc_count = np.zeros(n_corpus_docs)
        if has_doc_ids and all_doc_key_embeddings is not None:
            # Use doc_ids to index directly into the full corpus.
            for i in range(len(doc_ids)):
                for k in range(doc_ids.shape[1]):
                    d = int(doc_ids[i, k])
                    if d < n_corpus_docs:
                        doc_avg_logit[d] += per_doc_logits[i, k, c_idx]
                        doc_count[d] += 1
        else:
            # Map each retrieved key to its nearest unique-key index (vectorized).
            flat_keys = doc_key_emb.reshape(-1, doc_key_emb.shape[-1])  # (N*K, D)
            flat_logits = per_doc_logits[:, :, c_idx].reshape(-1)  # (N*K,)
            # Cosine-style: use dot product against normalized keys for speed.
            sims = flat_keys @ corpus_keys.T  # (N*K, n_unique)
            uids = sims.argmax(axis=1)  # (N*K,)
            for idx, uid in enumerate(uids):
                doc_avg_logit[uid] += flat_logits[idx]
                doc_count[uid] += 1

        retrieved_mask = doc_count > 0
        doc_avg_logit[retrieved_mask] /= doc_count[retrieved_mask]

        k_proj = _reduce_2d(corpus_keys, method)
        sc = ax2.scatter(
            k_proj[:, 0],
            k_proj[:, 1],
            c=doc_avg_logit,
            cmap="RdBu_r",
            s=100,
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )
        # Only label docs in small corpora.
        if n_corpus_docs <= 20:
            for i, (x, y) in enumerate(k_proj):
                label = f"D{i}" if retrieved_mask[i] else f"D{i}*"
                ax2.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=7)
        ax2.set_xlabel(f"{method_label}-1")
        ax2.set_ylabel(f"{method_label}-2")
        ax2.set_title(
            f"Doc Key Embeddings ({method_label})\n"
            "color = avg LOO Δlogit (red = pushes toward positive)"
        )
        fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    else:
        ax2.text(
            0.5,
            0.5,
            "doc_key_embeddings or\nper-doc Δlogit not available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=10,
        )
        ax2.set_title("Doc Key Embeddings (N/A)")

    # ------------------------------------------------------------------
    # Plot 3: Per-doc LOO Δlogit vs differentiable score, colored by patient class
    # ------------------------------------------------------------------
    ax3 = axes[0, 2]
    if has_per_doc:
        c_idx = min(1, per_doc_logits.shape[-1] - 1)
        n_patients, k_docs = diff_scores.shape

        x_vals = per_doc_logits[:, :, c_idx].reshape(-1)  # per-doc LOO Δlogit
        y_vals = diff_scores.reshape(-1)  # differentiable doc score
        colors = np.array(
            ["blue" if pos_mask[i] else "red" for i in range(n_patients) for _ in range(k_docs)]
        )

        ax3.scatter(x_vals, y_vals, c=colors, s=20, alpha=0.2, edgecolors="none")
        ax3.set_xlabel("Per-doc LOO Δlogit")
        ax3.set_ylabel("Differentiable doc score")
        ax3.set_title("Doc Contribution vs Retrieval Score")

        # Legend.
        ax3.scatter([], [], c="blue", label="Positive patient", s=40)
        ax3.scatter([], [], c="red", label="Negative patient", s=40)
        ax3.legend(fontsize=8)
    else:
        ax3.text(
            0.5,
            0.5,
            "per_doc_logits not available",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=10,
        )
        ax3.set_title("Doc Prediction vs Score (N/A)")

    # ------------------------------------------------------------------
    # Plot 4: Per-doc LOO Δlogit vs patient predicted logit, colored by doc score
    # ------------------------------------------------------------------
    ax4 = axes[1, 0]
    if has_per_doc:
        c_idx = min(1, per_doc_logits.shape[-1] - 1)
        n_patients, k_docs = diff_scores.shape

        # x: each doc's positive logit (per patient-doc pair)
        x_vals = per_doc_logits[:, :, c_idx].reshape(-1)
        # y: patient's final predicted positive logit (repeated for each of their k docs)
        patient_logit = logits[:, c_idx] if logits.shape[1] > 1 else logits[:, 0]
        y_vals = np.repeat(patient_logit, k_docs)
        # color: softmax-normalized doc score (contribution weight per patient)
        score_max = diff_scores.max(axis=1, keepdims=True)
        score_exp = np.exp(diff_scores - score_max)
        score_weights = score_exp / score_exp.sum(axis=1, keepdims=True)  # (N, K)
        color_vals = score_weights.reshape(-1)

        hb = ax4.hexbin(
            x_vals, y_vals, C=color_vals, reduce_C_function=np.mean, gridsize=30, cmap="viridis", mincnt=1
        )
        ax4.set_xlabel("Per-doc LOO Δlogit")
        ax4.set_ylabel("Patient predicted logit (positive)")
        ax4.set_title("Doc Δlogit vs Patient Logit\ncolor = avg doc weight (softmax)")
        fig.colorbar(hb, ax=ax4, fraction=0.046, pad=0.04, label="Avg doc weight")
    else:
        ax4.text(
            0.5,
            0.5,
            "per_doc_logits not available",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=10,
        )
        ax4.set_title("Doc vs Patient Logit (N/A)")

    # ------------------------------------------------------------------
    # Plot 5: Prediction summary (accuracy + AUROC)
    # ------------------------------------------------------------------
    ax5 = axes[1, 1]
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

    # Simple AUROC computation (no sklearn needed).
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

    bars = ax5.bar(range(len(metrics)), list(metrics.values()), color=["steelblue", "coral"][: len(metrics)])
    ax5.set_xticks(range(len(metrics)))
    ax5.set_xticklabels(list(metrics.keys()))
    ax5.set_ylim(0, 1.05)
    ax5.set_title("Prediction Summary")
    for bar, val in zip(bars, metrics.values(), strict=False):
        ax5.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # ------------------------------------------------------------------
    # Plot 6: Score-weighted average LOO Δlogit by class (box plot)
    # ------------------------------------------------------------------
    ax6 = axes[1, 2]
    if has_per_doc:
        c_idx = min(1, per_doc_logits.shape[-1] - 1)

        # softmax(differentiable_doc_scores) → weight per retrieved doc.
        score_max = diff_scores.max(axis=1, keepdims=True)
        score_exp = np.exp(diff_scores - score_max)
        score_weights = score_exp / score_exp.sum(axis=1, keepdims=True)  # (N, K)

        # Score-weighted average positive logit per patient.
        doc_pos_logits = per_doc_logits[:, :, c_idx]  # (N, K)
        weighted_avg_logit = (score_weights * doc_pos_logits).sum(axis=1)  # (N,)

        neg_vals = weighted_avg_logit[neg_mask]
        pos_vals = weighted_avg_logit[pos_mask]

        bp = ax6.boxplot(
            [neg_vals, pos_vals],
            labels=["Negative (0)", "Positive (1)"],
            patch_artist=True,
            widths=0.5,
        )
        bp["boxes"][0].set_facecolor("cornflowerblue")
        bp["boxes"][1].set_facecolor("salmon")
        ax6.set_xlabel("Ground truth label")
        ax6.set_ylabel("Score-weighted avg LOO Δlogit")
        ax6.set_title("Soft Retrieval Class-Conditionality (LOO)")
    else:
        ax6.text(
            0.5,
            0.5,
            "per_doc_logits not available",
            ha="center",
            va="center",
            transform=ax6.transAxes,
            fontsize=10,
        )
        ax6.set_title("Soft Retrieval Class-Conditionality (N/A)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic plots saved to {output_path}")


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
    artifacts, artifact_path, all_doc_keys, retrieval_db_path = run_extraction(run_dir)
    print(f"Artifacts saved to {artifact_path}")
    print(f"Keys: {sorted(artifacts.keys())}")
    for k, v in sorted(artifacts.items()):
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")

    all_doc_keys_np = all_doc_keys.numpy() if all_doc_keys is not None else None
    if all_doc_keys_np is not None:
        print(f"  Full corpus: {all_doc_keys_np.shape[0]} documents")

    method_outputs = {
        "pca": artifact_path.parent / "extraction_diagnostics.png",
        "tsne": artifact_path.parent / "extraction_diagnostics_tsne.png",
        "umap": artifact_path.parent / "extraction_diagnostics_umap.png",
    }
    for method, output_path in method_outputs.items():
        try:
            generate_plots(
                artifacts,
                output_path,
                all_doc_key_embeddings=all_doc_keys_np,
                method=method,
            )
        except ImportError as exc:
            print(
                f"Skipping {method} plot ({output_path.name}): {exc}. "
                f"Install the missing package to enable it.",
                file=sys.stderr,
            )

    if retrieval_db_path is not None:
        write_top_retrieved_docs(
            artifacts,
            artifact_path.parent / "top_retrieved_docs.csv",
            retrieval_db_path=retrieval_db_path,
            n_top=100,
        )
    else:
        print("Skipping top_retrieved_docs.csv: no retrieval DB path (InMemoryRetriever run).")


if __name__ == "__main__":
    main()
