"""Extract retrieval artifacts from a trained MedRAP model.

This module provides functions to run a trained model over a dataset and save
per-sample retrieval artifacts (document IDs, scores, query embeddings, key
embeddings, logits, and targets) to disk as a ``.pt`` file.

Output format
-------------
The saved ``.pt`` file contains a ``dict[str, Tensor]`` with keys:

=========  =====================  ========  ==============================
Key        Shape                  dtype     Present when
=========  =====================  ========  ==============================
logits     ``(N, C)``             float32   Always
targets    ``(N,)``               float32   When dataset has labels
query_embeddings                            Always
           ``(N, R, D_ret)``      float32
doc_ids    ``(N, R, K)``          int64     Retriever provides them
doc_scores ``(N, R, K)``          float32   Retriever provides them
doc_key_embeddings                          Retriever provides them
           ``(N, R, K, D_ret)``   float32
per_doc_logits                              marginalized_retrieval=True
           ``(N, K, C)``          float32
differentiable_doc_scores                   Either produced natively by
           ``(N, K)``             float32   marginalized_retrieval=True,
                                            or filled post-hoc from
                                            query_embeddings and
                                            doc_key_embeddings when both
                                            are present.
=========  =====================  ========  ==============================

``N`` is the total number of samples. Tensor position maps 1:1 to dataset
order (the dataloader must not shuffle).

Downstream usage examples
-------------------------

**Loading artifacts**::

    import torch

    artifacts = torch.load("extraction_artifacts.pt")

**Look up retrieved document text** (using ``doc_ids`` to index into the
retrieval HuggingFace dataset)::

    from datasets import load_from_disk

    retrieval_ds = load_from_disk("path/to/retrieval_dataset")
    sample_idx = 0
    doc_id = artifacts["doc_ids"][sample_idx, 0, 0].item()  # first query, first doc
    doc_text = retrieval_ds[doc_id]["doc_text"]

**Embedding visualization** (t-SNE / UMAP of query vs key embeddings)::

    query_embs = artifacts["query_embeddings"][:, 0, :]  # (N, D_ret)
    key_embs = artifacts["doc_key_embeddings"][:, 0, 0, :]  # (N, D_ret) first doc
    all_embs = torch.cat([query_embs, key_embs], dim=0)
    labels = ["query"] * len(query_embs) + ["key"] * len(key_embs)
    # ... pass all_embs.numpy() to sklearn.manifold.TSNE or umap.UMAP

**Human validation spreadsheet**::

    import pandas as pd

    df = pd.DataFrame(
        {
            "sample": range(len(artifacts["doc_ids"])),
            "top_doc_id": artifacts["doc_ids"][:, 0, 0].tolist(),
            "top_doc_score": artifacts["doc_scores"][:, 0, 0].tolist(),
        }
    )
    df.to_csv("retrieval_pairs.csv", index=False)
"""

from pathlib import Path

import lightning
import torch
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler


def _fill_differentiable_doc_scores(artifacts: dict[str, Tensor], *, similarity: str = "dot") -> None:
    """Compute ``differentiable_doc_scores`` post-hoc from saved query/key embeddings.

    Non-marginalized runs (e.g. cross-attention + ``BinaryClassificationLoss``)
    do not produce ``differentiable_doc_scores`` natively, but downstream
    diagnostics (`run_demographic_heatmap.py`, score-weighted plots) expect
    them. Since both ``query_embeddings`` and ``doc_key_embeddings`` are saved
    for every run that uses a retriever exposing keys, we can recover the
    score as their dot/cosine similarity.

    The artifacts dict is mutated in-place. If the key is already present, or
    either source tensor is missing, the dict is left unchanged.
    """
    from medrap.retrieval_scoring import differentiable_retrieval_scores

    if "differentiable_doc_scores" in artifacts:
        return
    if "query_embeddings" not in artifacts or "doc_key_embeddings" not in artifacts:
        return
    with torch.no_grad():
        scores = differentiable_retrieval_scores(
            artifacts["query_embeddings"],
            artifacts["doc_key_embeddings"],
            similarity=similarity,
        )
    artifacts["differentiable_doc_scores"] = scores.float().cpu()


def collate_prediction_batches(predictions: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Concatenate per-batch prediction dicts into a single dict of tensors.

    Takes the list of dicts returned by ``trainer.predict()`` and concatenates
    each key's tensors along dim=0. Only keys present in **all** batches are
    included in the output.

    Args:
        predictions: List of per-batch dicts, each mapping string keys to
            tensors with a leading batch dimension.

    Returns:
        Dict mapping each key to a single concatenated tensor of shape
        ``(N, ...)``.

    Examples:
        >>> batch_0 = {"logits": torch.tensor([[0.1], [0.2]]), "scores": torch.tensor([1.0, 2.0])}
        >>> batch_1 = {"logits": torch.tensor([[0.3]]), "scores": torch.tensor([3.0])}
        >>> result = collate_prediction_batches([batch_0, batch_1])
        >>> sorted(result.keys())
        ['logits', 'scores']
        >>> result["logits"].tolist()
        [[0.10000000149011612], [0.20000000298023224], [0.30000001192092896]]
        >>> result["scores"].tolist()
        [1.0, 2.0, 3.0]

    Keys missing from some batches are excluded:

        >>> batch_a = {"logits": torch.tensor([[1.0]]), "extra": torch.tensor([9.0])}
        >>> batch_b = {"logits": torch.tensor([[2.0]])}
        >>> result = collate_prediction_batches([batch_a, batch_b])
        >>> sorted(result.keys())
        ['logits']
    """
    if not predictions:
        return {}

    common_keys = set(predictions[0].keys())
    for batch in predictions[1:]:
        common_keys &= set(batch.keys())

    return {key: torch.cat([batch[key] for batch in predictions], dim=0) for key in sorted(common_keys)}


def extract_artifacts(
    module: lightning.LightningModule,
    dataloader: DataLoader,
    trainer: lightning.Trainer,
    *,
    output_dir: str | Path,
) -> Path:
    """Run prediction and save retrieval artifacts to disk.

    Calls ``trainer.predict(module, dataloaders=dataloader)``, collates the
    per-batch results, and saves them as a single ``.pt`` file.

    Args:
        module: Trained Lightning module with a ``predict_step`` that returns
            a dict of tensors (see
            :meth:`~medrap.lightning_module.MedRAPSupervisedLightningModule.predict_step`).
        dataloader: DataLoader to iterate over.
        trainer: Lightning Trainer to use for prediction.
        output_dir: Directory to save the artifacts into. Will be created if
            it does not exist.

    Returns:
        Path to the saved ``.pt`` file.

    Examples:
        >>> import tempfile
        >>> from medrap.encoders import MEDSCodeEncoder
        >>> from medrap.fusion import ReplaceFusion
        >>> from medrap.heads import LinearHead
        >>> from medrap.pooling import IdentityPooling
        >>> from medrap.query_projection import SequenceMeanQueryProjector
        >>> from medrap.retrieval_encoder import MeanPooledRetrievalEncoder
        >>> from medrap.retrievers import InMemoryRetriever
        >>> from medrap.model import RetrievalAugmentedModel
        >>> from medrap.lightning_module import MedRAPSupervisedLightningModule
        >>> model = RetrievalAugmentedModel(
        ...     encoder=MEDSCodeEncoder(),
        ...     query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
        ...     retriever=InMemoryRetriever(
        ...         doc_key_embeddings=torch.FloatTensor([[1, 0, 0, 0], [0, 1, 0, 0]]),
        ...         doc_tokens=torch.LongTensor([[1, 2], [3, 4]]),
        ...         doc_attention_mask=torch.BoolTensor([[True, True], [True, True]]),
        ...     ),
        ...     retrieval_encoder=MeanPooledRetrievalEncoder(vocab_size=8, embedding_dim=2),
        ...     fusion=ReplaceFusion(),
        ...     pooling=IdentityPooling(),
        ...     head=LinearHead(in_dim=2, out_dim=1),
        ... )
        >>> module = MedRAPSupervisedLightningModule(model=model)
        >>> batch = make_supervised_batch()
        >>> dl = torch.utils.data.DataLoader([batch], batch_size=None)
        >>> trainer = lightning.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = extract_artifacts(module, dl, trainer, output_dir=tmpdir)
        ...     artifacts = torch.load(path, weights_only=True)
        ...     sorted(artifacts.keys())
        ['differentiable_doc_scores', 'doc_ids', 'doc_key_embeddings', 'doc_scores', 'logits', 'query_embeddings', 'targets']
    """
    if isinstance(dataloader.sampler, RandomSampler):
        raise ValueError(
            "extract_artifacts requires shuffle=False: row i of the saved .pt must "
            "correspond to sample i of the dataset in dataloader-walk order."
        )
    num_devices = getattr(trainer, "num_devices", 1)
    if num_devices and num_devices > 1:
        raise ValueError(
            "extract_artifacts requires a single-device trainer; multi-device predict "
            f"can return rank-interleaved outputs (got num_devices={num_devices})."
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    batch_predictions = trainer.predict(module, dataloaders=dataloader)
    collated = collate_prediction_batches(batch_predictions)
    _fill_differentiable_doc_scores(collated)

    artifact_path = out / "extraction_artifacts.pt"
    torch.save(collated, artifact_path)
    return artifact_path
