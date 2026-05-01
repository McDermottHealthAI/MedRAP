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

from medrap.types import FusionInput


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
    use_cache: bool = False,
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
        use_cache: When ``True``, return the existing ``extraction_artifacts.pt``
            in ``output_dir`` without re-running ``trainer.predict``. Skips the
            ``shuffle`` and ``num_devices`` validation since the dataloader and
            trainer are not used in this branch. Defaults to ``False`` (always
            re-run, original behavior). Delete the ``.pt`` to force a recompute.

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
    out = Path(output_dir)
    artifact_path = out / "extraction_artifacts.pt"

    if use_cache and artifact_path.is_file():
        return artifact_path

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

    out.mkdir(parents=True, exist_ok=True)

    batch_predictions = trainer.predict(module, dataloaders=dataloader)
    collated = collate_prediction_batches(batch_predictions)
    _fill_differentiable_doc_scores(collated)

    torch.save(collated, artifact_path)
    return artifact_path


def compute_per_doc_logits_single_doc(
    module: lightning.LightningModule, dataloader: DataLoader
) -> Tensor:
    """Compute per-document logits via single-doc forward passes.

    For each retrieved document ``k`` per sample, re-run only the post-retrieval
    pipeline (``fusion -> pooling -> head``) with ``retrieval_memory`` sliced to
    that single doc. The resulting ``logits[i, k, :]`` answers: "what would the
    model predict for sample ``i`` if it only saw document ``k``?"

    This is the closest semantic substitute for the marginalized
    ``per_doc_logits`` when the model uses cross-attention fusion (which fuses
    all K docs jointly and produces no native per-doc prediction).

    Cost is roughly ``K`` extra forwards through ``fusion -> pooling -> head``
    per batch; ``encoder``, ``query_projector``, ``retriever``, and
    ``retrieval_encoder`` each run once per batch.

    Args:
        module: Trained Lightning module with a ``RetrievalAugmentedModel`` at
            ``module.model``. The retrieval encoder must produce
            ``retrieval_memory`` shaped ``(B, R, K, S_doc, D_mem)``.
        dataloader: DataLoader to iterate over. Must NOT shuffle, so the row
            order matches that of :func:`extract_artifacts`.

    Returns:
        Tensor with shape ``(N_total, K, C)`` and floating dtype, on CPU.

    Notes:
        Cross-attention models trained with ``K > 1`` see only one doc at a
        time during this computation -- the values are mildly out of training
        distribution. Treat them as relative per-doc rankings, not as
        calibrated probabilities. The pooling call mirrors
        ``RetrievalAugmentedModel.forward`` exactly (no attention mask passed),
        so any padding-related quirk in production is preserved.

    Examples:
        >>> from medrap.encoders import MEDSCodeEncoder
        >>> from medrap.fusion import CrossAttentionFusion
        >>> from medrap.heads import LinearHead
        >>> from medrap.lightning_module import MedRAPSupervisedLightningModule
        >>> from medrap.model import RetrievalAugmentedModel
        >>> from medrap.pooling import MaskedMeanPooling
        >>> from medrap.query_projection import SequenceMeanQueryProjector
        >>> from medrap.retrieval_encoder import TokenFeatureRetrievalEncoder
        >>> from medrap.retrievers import InMemoryRetriever
        >>> _ = torch.manual_seed(0)
        >>> model = RetrievalAugmentedModel(
        ...     encoder=MEDSCodeEncoder(),
        ...     query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
        ...     retriever=InMemoryRetriever(
        ...         doc_key_embeddings=torch.eye(4)[:3].float(),
        ...         doc_tokens=torch.LongTensor([[1, 2], [3, 4], [5, 6]]),
        ...         doc_attention_mask=torch.BoolTensor([[True, True]] * 3),
        ...         k=3,
        ...     ),
        ...     retrieval_encoder=TokenFeatureRetrievalEncoder(vocab_size=16, embedding_dim=4),
        ...     fusion=CrossAttentionFusion(
        ...         d_model=8, num_heads=2, ff_dim=16, num_layers=1, d_in_patient=1, d_in_doc=4
        ...     ),
        ...     pooling=MaskedMeanPooling(),
        ...     head=LinearHead(in_dim=8, out_dim=1),
        ... )
        >>> module = MedRAPSupervisedLightningModule(model=model)
        >>> dl = torch.utils.data.DataLoader([make_supervised_batch()], batch_size=None)
        >>> per_doc = compute_per_doc_logits_single_doc(module, dl)
        >>> tuple(per_doc.shape)
        (2, 3, 1)
    """
    if isinstance(dataloader.sampler, RandomSampler):
        raise ValueError(
            "compute_per_doc_logits_single_doc requires shuffle=False so row order "
            "matches extract_artifacts."
        )

    model = module.model
    device = next(model.parameters()).device
    was_training = module.training
    module.eval()

    per_batch: list[Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = module.transfer_batch_to_device(batch, device, dataloader_idx=0)
            encoder_out = model.encoder(batch)
            query_out = model.query_projector(encoder_out.patient_state)
            retrieval_out = model.retriever(query_out.query_embeddings)
            retrieval_encoded = model.retrieval_encoder(retrieval_out)

            rm = retrieval_encoded.retrieval_memory  # (B, R, K, S_doc, D_mem)
            if rm.ndim != 5:
                raise ValueError(
                    "compute_per_doc_logits_single_doc expects 5D retrieval_memory "
                    f"(B, R, K, S_doc, D_mem); got {tuple(rm.shape)}"
                )
            mask = retrieval_out.doc_attention_mask  # (B, R, K, S_doc) or None
            b, r, k_docs, s_doc, d_mem = rm.shape

            # Vectorize the K dim into the batch dim: one fusion call instead of
            # K small ones. Row b*K + k of the effective batch corresponds to
            # (sample b, doc k) -- the same pairing the original K-loop produced.
            rm_v = rm.transpose(1, 2).reshape(b * k_docs, r, s_doc, d_mem).unsqueeze(2)
            mask_v: Tensor | None = None
            if mask is not None:
                mask_v = mask.transpose(1, 2).reshape(b * k_docs, r, s_doc).unsqueeze(2)
            ps_v = encoder_out.patient_state.repeat_interleave(k_docs, dim=0)
            sid_v: Tensor | None = None
            if query_out.retrieval_step_ids is not None:
                sid_v = query_out.retrieval_step_ids.repeat_interleave(k_docs, dim=0)

            fused = model.fusion(
                FusionInput(
                    patient_state=ps_v,
                    retrieval_memory=rm_v,
                    retrieval_step_ids=sid_v,
                    doc_attention_mask=mask_v,
                )
            )
            pooled = model.pooling(fused.fused_state)
            logit = model.head(pooled)  # (B*K, C)
            per_batch.append(logit.view(b, k_docs, -1).detach().cpu())

    if was_training:
        module.train()

    return torch.cat(per_batch, dim=0).float()


def persist_single_doc_per_doc_logits(
    artifact_path: str | Path,
    module: lightning.LightningModule,
    dataloader: DataLoader,
) -> Tensor:
    """Compute and cache single-doc per-doc logits in the saved ``.pt`` artifact.

    On the first call this runs :func:`compute_per_doc_logits_single_doc`, loads
    the existing artifact dict from ``artifact_path``, adds a
    ``per_doc_logits_single_doc`` key, and re-saves. On subsequent calls the
    cached tensor is returned without invoking the model. To force a recompute,
    delete the ``.pt`` file (or the key) and call again.

    Args:
        artifact_path: Path to the ``.pt`` written by :func:`extract_artifacts`.
        module: Trained Lightning module (see
            :func:`compute_per_doc_logits_single_doc`).
        dataloader: Same ``shuffle=False`` dataloader used for extraction.

    Returns:
        The ``per_doc_logits_single_doc`` tensor with shape ``(N, K, C)``.
    """
    path = Path(artifact_path)
    artifacts: dict[str, Tensor] = torch.load(path, weights_only=True)

    cached = artifacts.get("per_doc_logits_single_doc")
    if cached is not None:
        return cached

    per_doc_logits = compute_per_doc_logits_single_doc(module, dataloader)
    artifacts["per_doc_logits_single_doc"] = per_doc_logits
    torch.save(artifacts, path)
    return per_doc_logits


def compute_per_doc_loo_delta_logits(
    module: lightning.LightningModule, dataloader: DataLoader
) -> Tensor:
    """Compute per-document Δlogit via leave-one-out forward passes.

    For each retrieved document ``k``, run the model with the other ``K-1`` docs
    as retrieval context, then take ``Δ[i, k] = full_logit[i] - leave_k_out_logit[i]``.
    A positive Δ means doc ``k`` pushed sample ``i`` toward the positive class
    relative to the model's prediction without it.

    Unlike :func:`compute_per_doc_logits_single_doc`, this stays in the training
    distribution (the cross-attention layer always sees a multi-doc context),
    so the per-doc signal is more meaningful when the model relies on
    cross-document interactions.

    Cost per batch is roughly **two** vectorized forwards through
    ``fusion -> pooling -> head`` (one full, one with effective batch ``B*K`` and
    one doc masked per row); ``encoder``, ``query_projector``, ``retriever``,
    and ``retrieval_encoder`` each run once per batch.

    Args:
        module: Trained Lightning module with a ``RetrievalAugmentedModel`` at
            ``module.model``. The retrieval encoder must produce
            ``retrieval_memory`` shaped ``(B, R, K, S_doc, D_mem)``.
        dataloader: DataLoader to iterate over. Must NOT shuffle, so the row
            order matches that of :func:`extract_artifacts`.

    Returns:
        Tensor with shape ``(N_total, K, C)`` and floating dtype, on CPU.

    Examples:
        >>> from medrap.encoders import MEDSCodeEncoder
        >>> from medrap.fusion import CrossAttentionFusion
        >>> from medrap.heads import LinearHead
        >>> from medrap.lightning_module import MedRAPSupervisedLightningModule
        >>> from medrap.model import RetrievalAugmentedModel
        >>> from medrap.pooling import MaskedMeanPooling
        >>> from medrap.query_projection import SequenceMeanQueryProjector
        >>> from medrap.retrieval_encoder import TokenFeatureRetrievalEncoder
        >>> from medrap.retrievers import InMemoryRetriever
        >>> _ = torch.manual_seed(0)
        >>> model = RetrievalAugmentedModel(
        ...     encoder=MEDSCodeEncoder(),
        ...     query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
        ...     retriever=InMemoryRetriever(
        ...         doc_key_embeddings=torch.eye(4)[:3].float(),
        ...         doc_tokens=torch.LongTensor([[1, 2], [3, 4], [5, 6]]),
        ...         doc_attention_mask=torch.BoolTensor([[True, True]] * 3),
        ...         k=3,
        ...     ),
        ...     retrieval_encoder=TokenFeatureRetrievalEncoder(vocab_size=16, embedding_dim=4),
        ...     fusion=CrossAttentionFusion(
        ...         d_model=8, num_heads=2, ff_dim=16, num_layers=1, d_in_patient=1, d_in_doc=4
        ...     ),
        ...     pooling=MaskedMeanPooling(),
        ...     head=LinearHead(in_dim=8, out_dim=1),
        ... )
        >>> module = MedRAPSupervisedLightningModule(model=model)
        >>> dl = torch.utils.data.DataLoader([make_supervised_batch()], batch_size=None)
        >>> delta = compute_per_doc_loo_delta_logits(module, dl)
        >>> tuple(delta.shape)
        (2, 3, 1)
    """
    if isinstance(dataloader.sampler, RandomSampler):
        raise ValueError(
            "compute_per_doc_loo_delta_logits requires shuffle=False so row order "
            "matches extract_artifacts."
        )

    model = module.model
    device = next(model.parameters()).device
    was_training = module.training
    module.eval()

    per_batch: list[Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = module.transfer_batch_to_device(batch, device, dataloader_idx=0)
            encoder_out = model.encoder(batch)
            query_out = model.query_projector(encoder_out.patient_state)
            retrieval_out = model.retriever(query_out.query_embeddings)
            retrieval_encoded = model.retrieval_encoder(retrieval_out)

            rm = retrieval_encoded.retrieval_memory  # (B, R, K, S_doc, D_mem)
            if rm.ndim != 5:
                raise ValueError(
                    "compute_per_doc_loo_delta_logits expects 5D retrieval_memory "
                    f"(B, R, K, S_doc, D_mem); got {tuple(rm.shape)}"
                )
            mask = retrieval_out.doc_attention_mask  # (B, R, K, S_doc) or None
            b, r, k_docs, s_doc, _ = rm.shape

            # 1. Full forward (all K docs visible).
            full_fused = model.fusion(
                FusionInput(
                    patient_state=encoder_out.patient_state,
                    retrieval_memory=rm,
                    retrieval_step_ids=query_out.retrieval_step_ids,
                    doc_attention_mask=mask,
                )
            )
            full_logit = model.head(model.pooling(full_fused.fused_state))  # (B, C)

            # 2. Vectorized LOO: row b*K+k carries sample b's full retrieval,
            # but with doc k masked out. Effective batch B*K.
            rm_v = rm.repeat_interleave(k_docs, dim=0)  # (B*K, R, K, S_doc, D_mem)
            ps_v = encoder_out.patient_state.repeat_interleave(k_docs, dim=0)
            sid_v: Tensor | None = None
            if query_out.retrieval_step_ids is not None:
                sid_v = query_out.retrieval_step_ids.repeat_interleave(k_docs, dim=0)

            # Build the keep-mask: True everywhere except (row b*K+k, :, k, :).
            keep = torch.ones(b * k_docs, r, k_docs, s_doc, dtype=torch.bool, device=rm.device)
            row_idx = torch.arange(b * k_docs, device=rm.device)
            k_idx = row_idx % k_docs
            keep[row_idx, :, k_idx, :] = False
            mask_v = (
                mask.repeat_interleave(k_docs, dim=0) & keep if mask is not None else keep
            )

            loo_fused = model.fusion(
                FusionInput(
                    patient_state=ps_v,
                    retrieval_memory=rm_v,
                    retrieval_step_ids=sid_v,
                    doc_attention_mask=mask_v,
                )
            )
            loo_logit = model.head(model.pooling(loo_fused.fused_state))  # (B*K, C)

            delta = full_logit.unsqueeze(1) - loo_logit.view(b, k_docs, -1)
            per_batch.append(delta.detach().cpu())

    if was_training:
        module.train()

    return torch.cat(per_batch, dim=0).float()


def persist_loo_per_doc_logits(
    artifact_path: str | Path,
    module: lightning.LightningModule,
    dataloader: DataLoader,
) -> Tensor:
    """Compute and cache LOO Δlogits in the saved ``.pt`` artifact.

    On the first call this runs :func:`compute_per_doc_loo_delta_logits`, loads
    the existing artifact dict from ``artifact_path``, adds a
    ``per_doc_loo_delta_logits`` key, and re-saves. On subsequent calls the
    cached tensor is returned without invoking the model. To force a recompute,
    delete the ``.pt`` file (or the key) and call again.

    Args:
        artifact_path: Path to the ``.pt`` written by :func:`extract_artifacts`.
        module: Trained Lightning module (see
            :func:`compute_per_doc_loo_delta_logits`).
        dataloader: Same ``shuffle=False`` dataloader used for extraction.

    Returns:
        The ``per_doc_loo_delta_logits`` tensor with shape ``(N, K, C)``.
    """
    path = Path(artifact_path)
    artifacts: dict[str, Tensor] = torch.load(path, weights_only=True)

    cached = artifacts.get("per_doc_loo_delta_logits")
    if cached is not None:
        return cached

    delta = compute_per_doc_loo_delta_logits(module, dataloader)
    artifacts["per_doc_loo_delta_logits"] = delta
    torch.save(artifacts, path)
    return delta
