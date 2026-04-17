"""One-off diagnostic: how collapsed is the retriever?"""
import torch
from collections import Counter

art = torch.load("outputs/mimic_run_retrieval_only/extraction/extraction_artifacts.pt", weights_only=True)
doc_ids = art["doc_ids"]  # (N, R, K)
scores = art["doc_scores"]  # (N, R, K)
targets = art["targets"].int()  # (N,)

N, R, K = doc_ids.shape
print(f"N={N} patients, R={R} queries, K={K} docs/query")

top1 = doc_ids[:, 0, 0]  # (N,)
print(f"\n=== Top-1 doc concentration ===")
print(f"Unique top-1 docs across {N} patients: {top1.unique().numel()}")
c = Counter(top1.tolist())
top_hits = c.most_common(10)
print(f"Top 10 most-frequent top-1 docs (doc_id, count):")
for doc_id, count in top_hits:
    print(f"  doc_id={doc_id:>6}  count={count:>5}  ({100*count/N:.1f}%)")

print(f"\n=== Top-1 per label ===")
for lbl in [0, 1]:
    mask = targets == lbl
    t1 = top1[mask]
    print(f"label={lbl} (n={mask.sum().item()}): {t1.unique().numel()} unique top-1 docs")
    top3 = Counter(t1.tolist()).most_common(3)
    print(f"  top 3: {top3}")

print(f"\n=== Top-K overlap within a patient ===")
topk_sets = [set(doc_ids[i, 0, :].tolist()) for i in range(N)]
unique_per_patient = [len(s) for s in topk_sets]
print(f"Avg unique docs in top-{K} per patient: {sum(unique_per_patient)/N:.2f}  (max possible = {K})")
print(f"Patients with all-distinct top-K: {sum(1 for u in unique_per_patient if u == K)} / {N}")

print(f"\n=== Union of top-K docs across all patients ===")
all_retrieved = set()
for s in topk_sets:
    all_retrieved |= s
print(f"Total unique docs retrieved anywhere in top-{K}: {len(all_retrieved)}")

print(f"\n=== Score dynamic range ===")
top1_scores = scores[:, 0, 0]
print(f"Top-1 score: min={top1_scores.min():.3f}  max={top1_scores.max():.3f}  mean={top1_scores.mean():.3f}")
if K > 1:
    gap = scores[:, 0, 0] - scores[:, 0, 1]
    print(f"Top1 - Top2 gap: min={gap.min():.4f}  max={gap.max():.4f}  mean={gap.mean():.4f}  median={gap.median():.4f}")
