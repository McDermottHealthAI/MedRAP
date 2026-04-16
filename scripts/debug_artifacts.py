"""Check if doc_ids / scores are identical across all patients."""

import numpy as np
import torch

a = torch.load("outputs/mimic_run_retrieval_only/extraction/extraction_artifacts.pt", weights_only=True)

doc_ids = a["doc_ids"].numpy()
print("doc_ids shape:", doc_ids.shape)
if doc_ids.ndim == 3:
    doc_ids = doc_ids[:, 0, :]
print("squeezed shape:", doc_ids.shape)
print("first 5 rows:")
print(doc_ids[:5])

all_same = np.all(doc_ids == doc_ids[0:1])
print(f"\nAll rows identical? {all_same}")
unique_rows = len(np.unique(doc_ids, axis=0))
print(f"Unique doc_id rows: {unique_rows} / {doc_ids.shape[0]}")

scores = a["differentiable_doc_scores"].numpy()
print(f"\ndiff_scores shape: {scores.shape}")
print("first 5 rows:")
print(scores[:5])
unique_score_rows = len(np.unique(scores, axis=0))
print(f"Unique score rows: {unique_score_rows} / {scores.shape[0]}")
