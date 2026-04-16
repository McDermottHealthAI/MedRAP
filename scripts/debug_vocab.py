"""Check how many unique codes appear in the val set."""

import torch
from numpy.linalg import norm

a = torch.load("outputs/mimic_run_retrieval_only/extraction/extraction_artifacts.pt", weights_only=True)
print("Keys:", sorted(a.keys()))

# Check query embeddings variance while we're here
qe = a["query_embeddings"].numpy()
print(f"\nQuery embeddings shape: {qe.shape}")
qe_std = qe.std(axis=0)
print(
    f"Query embedding std (per dim, across patients): "
    f"mean={qe_std.mean():.6f}, min={qe_std.min():.6f}, max={qe_std.max():.6f}"
)


# Cosine similarity between first patient and all others
q0 = qe[0, 0, :]  # (D,)
all_q = qe[:, 0, :]  # (N, D)
cos_sims = (all_q @ q0) / (norm(all_q, axis=1) * norm(q0) + 1e-8)
print(
    f"Cosine similarity to patient 0: "
    f"mean={cos_sims.mean():.4f}, min={cos_sims.min():.4f}, max={cos_sims.max():.4f}"
)
