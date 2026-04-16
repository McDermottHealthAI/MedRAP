"""Check how much score variance exists across patients for the same doc_ids."""

from collections import Counter

import numpy as np
import torch

a = torch.load("outputs/mimic_run_retrieval_only/extraction/extraction_artifacts.pt", weights_only=True)

doc_ids = a["doc_ids"].numpy()
if doc_ids.ndim == 3:
    doc_ids = doc_ids[:, 0, :]
scores = a["differentiable_doc_scores"].numpy()


# Softmax the scores like the heatmap code does
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


weights = softmax(scores)

print(f"Patients: {doc_ids.shape[0]}, K={doc_ids.shape[1]}")
print("\nSoftmax weights (first 10 patients):")
for i in range(min(10, len(weights))):
    w_str = ", ".join(f"{w:.4f}" for w in weights[i])
    d_str = ", ".join(str(d) for d in doc_ids[i])
    print(f"  patient {i}: docs=[{d_str}]  weights=[{w_str}]")

# How concentrated is weight on top-1 doc?
top1_weight = weights.max(axis=1)
print(
    f"\nTop-1 softmax weight: mean={top1_weight.mean():.4f}, "
    f"min={top1_weight.min():.4f}, max={top1_weight.max():.4f}"
)

# What fraction of patients share the same top-1 doc?
top1_docs = doc_ids[np.arange(len(doc_ids)), weights.argmax(axis=1)]
top1_freq = Counter(top1_docs.tolist())
print(f"\nUnique top-1 docs: {len(top1_freq)}")
print("Top-1 doc distribution:")
for did, cnt in top1_freq.most_common(10):
    print(f"  doc_id={did}: {cnt} patients ({100 * cnt / len(doc_ids):.1f}%)")
