"""Check how many unique codes appear in the val set."""
import torch

a = torch.load("outputs/mimic_run_retrieval_only/extraction/extraction_artifacts.pt", weights_only=True)
print("Keys:", sorted(a.keys()))

# Check query embeddings variance while we're here
qe = a["query_embeddings"].numpy()
print(f"\nQuery embeddings shape: {qe.shape}")
print(f"Query embedding std (per dim, across patients): mean={qe.std(axis=0).mean():.6f}, min={qe.std(axis=0).min():.6f}, max={qe.std(axis=0).max():.6f}")

import numpy as np
# Cosine similarity between first patient and all others
from numpy.linalg import norm
q0 = qe[0, 0, :]  # (D,)
all_q = qe[:, 0, :]  # (N, D)
cos_sims = (all_q @ q0) / (norm(all_q, axis=1) * norm(q0) + 1e-8)
print(f"Cosine similarity to patient 0: mean={cos_sims.mean():.4f}, min={cos_sims.min():.4f}, max={cos_sims.max():.4f}")
