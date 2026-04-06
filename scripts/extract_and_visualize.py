"""Extract retrieval artifacts from a trained MedRAP run and generate diagnostic plots.

Loads the saved config and checkpoint from a training run directory, runs extraction
via ``extract_artifacts()``, and produces a 5-panel diagnostic figure.

Usage::

    python scripts/extract_and_visualize.py --run_dir outputs/margin_check_XXX

The script also works on real-data runs (e.g. MIMIC retrieval-only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightning
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader

# Ensure the project is importable when run from the repo root.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from medrap.configs import instantiate_datamodule, instantiate_training_module
from medrap.extraction import extract_artifacts


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


def run_extraction(run_dir: Path) -> tuple[dict[str, Tensor], Path, Tensor | None, Tensor | None]:
    """Load a trained model and extract artifacts from the val split.

    Returns (artifacts, artifact_path, all_doc_key_embeddings, code_means).
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

    # Datamodule — val split (deterministic, no shuffle).
    datamodule = instantiate_datamodule(cfg)
    dataloader: DataLoader = datamodule.val_dataloader()

    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    # Grab raw code values from the first batch (for scatter plot).
    first_batch = next(iter(dataloader))
    code_means: Tensor | None = None
    if hasattr(first_batch, "code"):
        code_means = first_batch.code.float().mean(dim=-1)  # (N,)

    extract_dir = run_dir / "extraction"
    artifact_path = extract_artifacts(module, dataloader, trainer, output_dir=extract_dir)
    artifacts = torch.load(artifact_path, weights_only=True)
    return artifacts, artifact_path, all_doc_keys, code_means


# ---------------------------------------------------------------------------
# PCA helper (avoids sklearn dependency)
# ---------------------------------------------------------------------------


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """Project rows of *X* to 2D via PCA (mean-centered SVD)."""
    X_centered = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:2].T


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def generate_plots(
    artifacts: dict[str, Tensor],
    output_path: Path,
    all_doc_key_embeddings: np.ndarray | None = None,
    code_means: np.ndarray | None = None,
) -> None:
    """Create a 6-panel diagnostic figure and save to *output_path*.

    Args:
        all_doc_key_embeddings: Full corpus key matrix ``(N_docs, D)`` from the
            retriever.  When provided, Plot 1 shows similarity against *all* docs
            rather than only the retrieved subset.
        code_means: Per-patient mean code value ``(N,)`` for the retrieval
            class-conditionality scatter plot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("MedRAP Retrieval Extraction Diagnostics", fontsize=14, y=0.98)

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
    q_proj = _pca_2d(query_emb)
    ax1.scatter(q_proj[neg_mask, 0], q_proj[neg_mask, 1], c="red", marker="o",
                alpha=0.3, s=40, edgecolors="none", label="Label 0")
    ax1.scatter(q_proj[pos_mask, 0], q_proj[pos_mask, 1], c="blue", marker="o",
                alpha=0.3, s=40, edgecolors="none", label="Label 1")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("Query Embeddings (PCA)")
    ax1.legend(fontsize=8)

    # Precompute n_docs_from_ids for later plots.
    if has_doc_ids:
        n_docs_from_ids = int(doc_ids.max()) + 1

    # ------------------------------------------------------------------
    # Plot 2: Doc key embedding PCA scatter, colored by avg per-doc logit
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
            flat_logits = per_doc_logits[:, :, c_idx].reshape(-1)       # (N*K,)
            # Cosine-style: use dot product against normalized keys for speed.
            sims = flat_keys @ corpus_keys.T  # (N*K, n_unique)
            uids = sims.argmax(axis=1)        # (N*K,)
            for idx, uid in enumerate(uids):
                doc_avg_logit[uid] += flat_logits[idx]
                doc_count[uid] += 1

        retrieved_mask = doc_count > 0
        doc_avg_logit[retrieved_mask] /= doc_count[retrieved_mask]

        k_proj = _pca_2d(corpus_keys)
        sc = ax2.scatter(k_proj[:, 0], k_proj[:, 1], c=doc_avg_logit, cmap="RdBu_r",
                         s=100, edgecolors="black", linewidths=1.0, zorder=5)
        # Only label docs in small corpora.
        if n_corpus_docs <= 20:
            for i, (x, y) in enumerate(k_proj):
                label = f"D{i}" if retrieved_mask[i] else f"D{i}*"
                ax2.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=7)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title("Doc Key Embeddings (PCA)\ncolor = avg positive logit")
        fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    else:
        ax2.text(0.5, 0.5, "doc_key_embeddings or\nper_doc_logits not available",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10)
        ax2.set_title("Doc Key Embeddings (N/A)")

    # ------------------------------------------------------------------
    # Plot 3: Per-doc logit vs differentiable score, colored by patient class
    # ------------------------------------------------------------------
    ax3 = axes[0, 2]
    if has_per_doc:
        c_idx = min(1, per_doc_logits.shape[-1] - 1)
        n_patients, k_docs = diff_scores.shape

        x_vals = per_doc_logits[:, :, c_idx].reshape(-1)  # per-doc positive logit
        y_vals = diff_scores.reshape(-1)                    # differentiable doc score
        colors = np.array([
            "blue" if pos_mask[i] else "red"
            for i in range(n_patients)
            for _ in range(k_docs)
        ])

        ax3.scatter(x_vals, y_vals, c=colors, s=20, alpha=0.2, edgecolors="none")
        ax3.set_xlabel("Per-doc positive logit")
        ax3.set_ylabel("Differentiable doc score")
        ax3.set_title("Doc Prediction vs Relevance Score")

        # Legend.
        ax3.scatter([], [], c="blue", label="Positive patient", s=40)
        ax3.scatter([], [], c="red", label="Negative patient", s=40)
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "per_doc_logits not available", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=10)
        ax3.set_title("Doc Prediction vs Score (N/A)")

    # ------------------------------------------------------------------
    # Plot 4: Per-doc positive logit vs patient predicted logit, colored by doc score
    # ------------------------------------------------------------------
    ax4 = axes[1, 0]
    if has_per_doc:
        c_idx = min(1, per_doc_logits.shape[-1] - 1)
        n_patients, k_docs = diff_scores.shape

        # x: each doc's positive logit (per patient-doc pair)
        x_vals = per_doc_logits[:, :, c_idx].reshape(-1)
        # y: per-doc positive logit (what this specific doc predicts for this patient)
        y_vals = per_doc_logits[:, :, c_idx].reshape(-1)
        # color: softmax-normalized doc score (contribution weight per patient)
        score_max = diff_scores.max(axis=1, keepdims=True)
        score_exp = np.exp(diff_scores - score_max)
        score_weights = score_exp / score_exp.sum(axis=1, keepdims=True)  # (N, K)
        color_vals = score_weights.reshape(-1)

        sc4 = ax4.scatter(x_vals, y_vals, c=color_vals, cmap="viridis", s=20, alpha=0.3,
                          edgecolors="none")
        ax4.set_xlabel("Per-doc positive logit (avg per doc)")
        ax4.set_ylabel("Per-doc positive logit (per patient-doc)")
        ax4.set_title("Doc Avg Logit vs Instance Logit\ncolor = doc weight (softmax)")
        fig.colorbar(sc4, ax=ax4, fraction=0.046, pad=0.04, label="Doc weight")
    else:
        ax4.text(0.5, 0.5, "per_doc_logits not available", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=10)
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

    bars = ax5.bar(range(len(metrics)), list(metrics.values()), color=["steelblue", "coral"][:len(metrics)])
    ax5.set_xticks(range(len(metrics)))
    ax5.set_xticklabels(list(metrics.keys()))
    ax5.set_ylim(0, 1.05)
    ax5.set_title("Prediction Summary")
    for bar, val in zip(bars, metrics.values()):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    # ------------------------------------------------------------------
    # Plot 6: Score-weighted average positive logit by class (box plot)
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
        ax6.set_ylabel("Score-weighted avg positive logit")
        ax6.set_title("Soft Retrieval Class-Conditionality")
    else:
        ax6.text(0.5, 0.5, "per_doc_logits not available", ha="center", va="center",
                 transform=ax6.transAxes, fontsize=10)
        ax6.set_title("Soft Retrieval Class-Conditionality (N/A)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic plots saved to {output_path}")


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
    artifacts, artifact_path, all_doc_keys, code_means = run_extraction(run_dir)
    print(f"Artifacts saved to {artifact_path}")
    print(f"Keys: {sorted(artifacts.keys())}")
    for k, v in sorted(artifacts.items()):
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")

    all_doc_keys_np = all_doc_keys.numpy() if all_doc_keys is not None else None
    code_means_np = code_means.numpy() if code_means is not None else None
    if all_doc_keys_np is not None:
        print(f"  Full corpus: {all_doc_keys_np.shape[0]} documents")

    output_path = artifact_path.parent / "extraction_diagnostics.png"
    generate_plots(artifacts, output_path, all_doc_key_embeddings=all_doc_keys_np, code_means=code_means_np)


if __name__ == "__main__":
    main()
