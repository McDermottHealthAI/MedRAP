import lightning
import pytest
import torch
from torch.utils.data import DataLoader

from conftest import make_supervised_batch
from medrap.encoders import MEDSCodeEncoder
from medrap.extraction import collate_prediction_batches, extract_artifacts
from medrap.fusion import CrossAttentionFusion, ReplaceFusion
from medrap.heads import LinearHead
from medrap.lightning_module import MedRAPSupervisedLightningModule
from medrap.losses import MarginalizedRetrievalSupervisedLoss
from medrap.model import RetrievalAugmentedModel
from medrap.pooling import IdentityPooling, MaskedMeanPooling
from medrap.query_projection import SequenceMeanQueryProjector
from medrap.retrieval_encoder import (
    KeyEmbeddingRetrievalEncoder,
    MeanPooledRetrievalEncoder,
    TokenFeatureRetrievalEncoder,
)
from medrap.retrieval_scoring import differentiable_retrieval_scores
from medrap.retrievers import InMemoryRetriever
from medrap.task import MarginalizedBinaryClassificationTask
from medrap.types import FusionInput


def _make_module() -> MedRAPSupervisedLightningModule:
    model = RetrievalAugmentedModel(
        encoder=MEDSCodeEncoder(),
        query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
        retriever=InMemoryRetriever(
            doc_key_embeddings=torch.FloatTensor([[1, 0, 0, 0], [0, 1, 0, 0]]),
            doc_tokens=torch.LongTensor([[1, 2], [3, 4]]),
            doc_attention_mask=torch.BoolTensor([[True, True], [True, True]]),
        ),
        retrieval_encoder=MeanPooledRetrievalEncoder(vocab_size=8, embedding_dim=2),
        fusion=ReplaceFusion(),
        pooling=IdentityPooling(),
        head=LinearHead(in_dim=2, out_dim=1),
    )
    return MedRAPSupervisedLightningModule(model=model)


def test_predict_step_returns_expected_keys() -> None:
    module = _make_module()
    batch = make_supervised_batch()
    result = module.predict_step(batch, batch_idx=0)

    assert "logits" in result
    assert "targets" in result
    assert "query_embeddings" in result
    assert "doc_ids" in result
    assert "doc_scores" in result
    assert "doc_key_embeddings" in result

    assert result["logits"].shape == (2, 1)
    assert result["targets"].shape == (2,)
    assert result["query_embeddings"].shape == (2, 1, 4)
    # K=1 (default k for InMemoryRetriever), R=1 (tabular mode)
    assert result["doc_ids"].shape == (2, 1, 1)
    assert result["doc_scores"].shape == (2, 1, 1)
    assert result["doc_key_embeddings"].shape == (2, 1, 1, 4)

    for tensor in result.values():
        assert tensor.device.type == "cpu"


def test_collate_prediction_batches() -> None:
    batch_0 = {"logits": torch.tensor([[0.1], [0.2]]), "scores": torch.tensor([1.0, 2.0])}
    batch_1 = {"logits": torch.tensor([[0.3]]), "scores": torch.tensor([3.0])}

    result = collate_prediction_batches([batch_0, batch_1])

    assert sorted(result.keys()) == ["logits", "scores"]
    assert result["logits"].shape == (3, 1)
    assert result["scores"].shape == (3,)


def test_collate_handles_missing_keys() -> None:
    batch_a = {"logits": torch.tensor([[1.0]]), "extra": torch.tensor([9.0])}
    batch_b = {"logits": torch.tensor([[2.0]])}

    result = collate_prediction_batches([batch_a, batch_b])

    assert sorted(result.keys()) == ["logits"]


def test_collate_empty_list() -> None:
    assert collate_prediction_batches([]) == {}


def test_extract_artifacts_end_to_end(tmp_path) -> None:
    module = _make_module()
    batch = make_supervised_batch()
    dl = DataLoader([batch], batch_size=None)
    trainer = lightning.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)

    path = extract_artifacts(module, dl, trainer, output_dir=tmp_path / "artifacts")

    assert path.exists()
    assert path.name == "extraction_artifacts.pt"

    artifacts = torch.load(path, weights_only=True)
    assert "logits" in artifacts
    assert "query_embeddings" in artifacts
    assert "doc_ids" in artifacts
    assert "doc_scores" in artifacts
    assert "doc_key_embeddings" in artifacts
    assert artifacts["logits"].shape[0] == 2


def test_extract_artifacts_rejects_shuffled_dataloader(tmp_path) -> None:
    """Shuffled loader would break row <-> sample alignment; must fail loud."""
    module = _make_module()
    batch = make_supervised_batch()
    dl = DataLoader([batch, batch], batch_size=None, shuffle=True)
    trainer = lightning.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)

    with pytest.raises(ValueError, match="shuffle"):
        extract_artifacts(module, dl, trainer, output_dir=tmp_path / "artifacts")


def test_extract_artifacts_rejects_multi_device_trainer(tmp_path, monkeypatch) -> None:
    """Multi-device predict can return rank-interleaved outputs; must fail loud."""
    module = _make_module()
    batch = make_supervised_batch()
    dl = DataLoader([batch], batch_size=None)
    trainer = lightning.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)
    monkeypatch.setattr(type(trainer), "num_devices", property(lambda self: 2))

    with pytest.raises(ValueError, match="device"):
        extract_artifacts(module, dl, trainer, output_dir=tmp_path / "artifacts")


def _make_marginalized_module() -> MedRAPSupervisedLightningModule:
    model = RetrievalAugmentedModel(
        encoder=MEDSCodeEncoder(),
        query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
        retriever=InMemoryRetriever(
            doc_key_embeddings=torch.FloatTensor([[1, 0, 0, 0], [0, 1, 0, 0]]),
            doc_tokens=torch.LongTensor([[1, 2], [3, 4]]),
            doc_attention_mask=torch.BoolTensor([[True, True], [True, True]]),
            k=2,
            similarity="dot",
        ),
        retrieval_encoder=KeyEmbeddingRetrievalEncoder(),
        fusion=ReplaceFusion(),
        pooling=IdentityPooling(),
        head=LinearHead(in_dim=4, out_dim=2),
        marginalized_retrieval=True,
    )
    return MedRAPSupervisedLightningModule(
        model=model,
        task=MarginalizedBinaryClassificationTask(),
        loss_fn=MarginalizedRetrievalSupervisedLoss(),
    )


def test_predict_step_values_match_pipeline_stages() -> None:
    """Every extracted tensor must equal the output of the corresponding pipeline stage."""
    module = _make_module()
    module.eval()
    batch = make_supervised_batch()

    with torch.no_grad():
        ref_encoded = module.model.encoder(batch)
        ref_query_out = module.model.query_projector(ref_encoded.patient_state)
        ref_retriever_out = module.model.retriever(ref_query_out.query_embeddings)
        ref_full = module.model(batch)

    result = module.predict_step(batch, batch_idx=0)

    assert torch.equal(result["logits"], ref_full.logits.detach().cpu())
    assert torch.equal(
        result["query_embeddings"], ref_query_out.query_embeddings.detach().cpu()
    )
    assert torch.equal(result["doc_ids"], ref_retriever_out.doc_ids.detach().cpu())
    assert torch.equal(result["doc_scores"], ref_retriever_out.doc_scores.detach().cpu())
    assert torch.equal(
        result["doc_key_embeddings"],
        ref_retriever_out.doc_key_embeddings.detach().cpu(),
    )
    assert torch.equal(result["targets"], batch.boolean_value.float().detach().cpu())

    # Structural invariant: doc_key_embeddings[i,r,k,:] is the corpus key at doc_ids[i,r,k].
    corpus_keys = module.model.retriever._doc_key_embeddings
    flat_ids = result["doc_ids"].reshape(-1)
    expected_keys = corpus_keys[flat_ids].reshape(result["doc_key_embeddings"].shape)
    assert torch.equal(result["doc_key_embeddings"], expected_keys)


def test_predict_step_values_match_pipeline_stages_marginalized() -> None:
    """Marginalized path: per_doc_logits and differentiable_doc_scores must match stages."""
    module = _make_marginalized_module()
    module.eval()
    batch = make_supervised_batch()

    with torch.no_grad():
        ref_encoded = module.model.encoder(batch)
        ref_query_out = module.model.query_projector(ref_encoded.patient_state)
        ref_retriever_out = module.model.retriever(ref_query_out.query_embeddings)
        ref_retrieval_encoded = module.model.retrieval_encoder(ref_retriever_out)
        ref_fusion_out = module.model.fusion(
            FusionInput(
                patient_state=ref_encoded.patient_state,
                retrieval_memory=ref_retrieval_encoded.retrieval_memory,
                retrieval_step_ids=ref_query_out.retrieval_step_ids,
                doc_attention_mask=ref_retriever_out.doc_attention_mask,
            )
        )
        ref_full = module.model(batch)

        b, k_docs, d_mem = ref_fusion_out.fused_state.shape
        num_classes = module.model.head.linear.out_features
        ref_per_doc_logits = module.model.head(
            ref_fusion_out.fused_state.reshape(-1, d_mem)
        ).view(b, k_docs, num_classes)
        ref_diff_scores = differentiable_retrieval_scores(
            ref_query_out.query_embeddings,
            ref_retriever_out.doc_key_embeddings,
            similarity=module.model.marginalized_score_similarity,
        )

    result = module.predict_step(batch, batch_idx=0)

    assert torch.equal(result["logits"], ref_full.logits.detach().cpu())
    assert torch.equal(
        result["query_embeddings"], ref_query_out.query_embeddings.detach().cpu()
    )
    assert torch.equal(result["doc_ids"], ref_retriever_out.doc_ids.detach().cpu())
    assert torch.equal(result["doc_scores"], ref_retriever_out.doc_scores.detach().cpu())
    assert torch.equal(
        result["doc_key_embeddings"],
        ref_retriever_out.doc_key_embeddings.detach().cpu(),
    )
    assert torch.equal(result["targets"], batch.boolean_value.float().detach().cpu())
    assert torch.equal(
        result["per_doc_logits"], ref_per_doc_logits.detach().cpu()
    )
    assert torch.equal(
        result["differentiable_doc_scores"], ref_diff_scores.detach().cpu()
    )


def test_extract_artifacts_fills_differentiable_doc_scores_for_non_marginalized(tmp_path) -> None:
    """Non-marginalized runs must get differentiable_doc_scores computed post-hoc.

    Cross-attention / replace-fusion runs without marginalized retrieval don't
    produce differentiable_doc_scores natively, but downstream diagnostics
    (keyword_demographic_heatmap, doc-score plots) need them. Since both
    query_embeddings and doc_key_embeddings are saved, we can compute it.
    """
    module = _make_module()  # non-marginalized
    batch = make_supervised_batch()
    dl = DataLoader([batch], batch_size=None)
    trainer = lightning.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)

    path = extract_artifacts(module, dl, trainer, output_dir=tmp_path / "artifacts")
    artifacts = torch.load(path, weights_only=True)

    assert "differentiable_doc_scores" in artifacts

    expected = differentiable_retrieval_scores(
        artifacts["query_embeddings"],
        artifacts["doc_key_embeddings"],
        similarity="dot",
    )
    torch.testing.assert_close(artifacts["differentiable_doc_scores"], expected)


def test_extract_artifacts_preserves_native_differentiable_doc_scores(tmp_path) -> None:
    """Marginalized runs produce differentiable_doc_scores natively; must not be overwritten."""
    module = _make_marginalized_module()
    batch = make_supervised_batch()
    dl = DataLoader([batch], batch_size=None)
    trainer = lightning.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)

    path = extract_artifacts(module, dl, trainer, output_dir=tmp_path / "artifacts")
    artifacts = torch.load(path, weights_only=True)

    module.eval()
    native = module.predict_step(batch, batch_idx=0)["differentiable_doc_scores"]
    torch.testing.assert_close(artifacts["differentiable_doc_scores"], native)


def test_fill_differentiable_doc_scores_is_noop_when_source_tensors_missing() -> None:
    """Helper must not add the key if query_embeddings or doc_key_embeddings is absent."""
    from medrap.extraction import _fill_differentiable_doc_scores

    artifacts = {
        "query_embeddings": torch.randn(2, 1, 4),
        "logits": torch.zeros(2, 1),
    }
    _fill_differentiable_doc_scores(artifacts)
    assert "differentiable_doc_scores" not in artifacts

    artifacts = {
        "doc_key_embeddings": torch.randn(2, 1, 3, 4),
        "logits": torch.zeros(2, 1),
    }
    _fill_differentiable_doc_scores(artifacts)
    assert "differentiable_doc_scores" not in artifacts


def test_extract_artifacts_use_cache_skips_predict_when_pt_exists(tmp_path) -> None:
    """When ``use_cache=True`` and the .pt already exists, return the path without running predict."""
    out = tmp_path / "artifacts"
    out.mkdir()
    artifact_path = out / "extraction_artifacts.pt"
    sentinel = {"sentinel_key": torch.tensor([42.0])}
    torch.save(sentinel, artifact_path)

    module = _make_module()
    batch = make_supervised_batch()
    dl = DataLoader([batch], batch_size=None)
    trainer = lightning.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)

    returned = extract_artifacts(module, dl, trainer, output_dir=out, use_cache=True)

    assert returned == artifact_path
    cached = torch.load(returned, weights_only=True)
    assert "sentinel_key" in cached
    assert "logits" not in cached


def test_extract_artifacts_use_cache_runs_when_pt_missing(tmp_path) -> None:
    """When ``use_cache=True`` and the .pt is absent, the function must run predict normally."""
    out = tmp_path / "artifacts"
    module = _make_module()
    batch = make_supervised_batch()
    dl = DataLoader([batch], batch_size=None)
    trainer = lightning.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)

    returned = extract_artifacts(module, dl, trainer, output_dir=out, use_cache=True)

    assert returned.is_file()
    artifacts = torch.load(returned, weights_only=True)
    assert "logits" in artifacts


def test_extract_artifacts_default_overwrites_existing_pt(tmp_path) -> None:
    """Default (``use_cache=False``) preserves old behavior: always re-run and overwrite."""
    out = tmp_path / "artifacts"
    out.mkdir()
    artifact_path = out / "extraction_artifacts.pt"
    sentinel = {"sentinel_key": torch.tensor([42.0])}
    torch.save(sentinel, artifact_path)

    module = _make_module()
    batch = make_supervised_batch()
    dl = DataLoader([batch], batch_size=None)
    trainer = lightning.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)

    extract_artifacts(module, dl, trainer, output_dir=out)
    artifacts = torch.load(artifact_path, weights_only=True)
    assert "sentinel_key" not in artifacts
    assert "logits" in artifacts


def _make_cross_attention_module(*, k: int = 3) -> MedRAPSupervisedLightningModule:
    """Tiny non-marginalized cross-attention module for single-doc forward tests."""
    d_in_patient = 1  # MEDSCodeEncoder produces patient_state with last-dim 1
    d_in_doc = 4  # TokenFeatureRetrievalEncoder embedding_dim
    d_model = 8
    corpus_keys = torch.eye(4)[:k].float()  # k docs, 4-dim keys
    corpus_tokens = torch.LongTensor([[i + 1, i + 2] for i in range(k)])
    corpus_mask = torch.BoolTensor([[True, True]] * k)

    model = RetrievalAugmentedModel(
        encoder=MEDSCodeEncoder(),
        query_projector=SequenceMeanQueryProjector(in_dim=d_in_patient, out_dim=4),
        retriever=InMemoryRetriever(
            doc_key_embeddings=corpus_keys,
            doc_tokens=corpus_tokens,
            doc_attention_mask=corpus_mask,
            k=k,
            similarity="dot",
        ),
        retrieval_encoder=TokenFeatureRetrievalEncoder(vocab_size=16, embedding_dim=d_in_doc),
        fusion=CrossAttentionFusion(
            d_model=d_model,
            num_heads=2,
            ff_dim=16,
            num_layers=1,
            d_in_patient=d_in_patient,
            d_in_doc=d_in_doc,
        ),
        pooling=MaskedMeanPooling(),
        head=LinearHead(in_dim=d_model, out_dim=1),
    )
    return MedRAPSupervisedLightningModule(model=model)


def test_compute_per_doc_logits_single_doc_shape_and_variation(tmp_path) -> None:
    """Helper must return (N, K, C) and produce different logits across the K axis.

    Without correct slicing every doc would see the same retrieval_memory and
    the K-axis variance would be zero.
    """
    from medrap.extraction import compute_per_doc_logits_single_doc

    torch.manual_seed(0)
    k_docs = 3
    module = _make_cross_attention_module(k=k_docs)
    batch_a = make_supervised_batch()
    batch_b = make_supervised_batch()
    dl = DataLoader([batch_a, batch_b], batch_size=None)

    per_doc_logits = compute_per_doc_logits_single_doc(module, dl)

    n_total = batch_a.code.shape[0] + batch_b.code.shape[0]
    assert per_doc_logits.shape == (n_total, k_docs, 1)
    assert per_doc_logits.dtype.is_floating_point
    assert per_doc_logits.std(dim=1).max().item() > 0.0


def test_compute_per_doc_logits_single_doc_matches_manual_slice() -> None:
    """For doc k, the helper's logit must equal a manual fusion(slice)+pool+head call."""
    from medrap.extraction import compute_per_doc_logits_single_doc

    torch.manual_seed(0)
    k_docs = 3
    module = _make_cross_attention_module(k=k_docs)
    module.eval()
    batch = make_supervised_batch()
    dl = DataLoader([batch], batch_size=None)

    per_doc_logits = compute_per_doc_logits_single_doc(module, dl)

    with torch.no_grad():
        encoder_out = module.model.encoder(batch)
        query_out = module.model.query_projector(encoder_out.patient_state)
        retriever_out = module.model.retriever(query_out.query_embeddings)
        retrieval_encoded = module.model.retrieval_encoder(retriever_out)
        rm = retrieval_encoded.retrieval_memory  # (B, R, K, S_doc, D_mem)
        mask = retriever_out.doc_attention_mask  # (B, R, K, S_doc) or None

        for k in range(k_docs):
            rm_k = rm[:, :, k : k + 1, :, :]
            mask_k = mask[:, :, k : k + 1, :] if mask is not None else None
            fused = module.model.fusion(
                FusionInput(
                    patient_state=encoder_out.patient_state,
                    retrieval_memory=rm_k,
                    retrieval_step_ids=query_out.retrieval_step_ids,
                    doc_attention_mask=mask_k,
                )
            )
            pooled = module.model.pooling(fused.fused_state)
            expected = module.model.head(pooled)
            torch.testing.assert_close(
                per_doc_logits[:, k, :], expected.detach().cpu()
            )


def test_persist_single_doc_per_doc_logits_creates_key_in_artifact(tmp_path) -> None:
    """First call must run the model and persist `per_doc_logits_single_doc` into the .pt."""
    from medrap.extraction import persist_single_doc_per_doc_logits

    torch.manual_seed(0)
    module = _make_cross_attention_module(k=3)
    batch = make_supervised_batch()
    dl = DataLoader([batch], batch_size=None)
    trainer = lightning.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)

    artifact_path = extract_artifacts(module, dl, trainer, output_dir=tmp_path / "artifacts")
    pre = torch.load(artifact_path, weights_only=True)
    assert "per_doc_logits_single_doc" not in pre
    assert "per_doc_logits" not in pre  # cross-attention path doesn't emit it natively

    pdl = persist_single_doc_per_doc_logits(artifact_path, module, dl)

    post = torch.load(artifact_path, weights_only=True)
    assert "per_doc_logits_single_doc" in post
    assert torch.equal(post["per_doc_logits_single_doc"], pdl)
    assert pdl.shape == (batch.code.shape[0], 3, 1)
    # Sibling keys must be preserved unchanged.
    for key in pre:
        assert torch.equal(post[key], pre[key])


def test_persist_single_doc_per_doc_logits_is_idempotent(tmp_path) -> None:
    """Second call must reuse the cached tensor and not re-invoke the model."""
    from medrap import extraction as extraction_mod
    from medrap.extraction import persist_single_doc_per_doc_logits

    torch.manual_seed(0)
    module = _make_cross_attention_module(k=3)
    batch = make_supervised_batch()
    dl = DataLoader([batch], batch_size=None)
    trainer = lightning.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)
    artifact_path = extract_artifacts(module, dl, trainer, output_dir=tmp_path / "artifacts")

    first = persist_single_doc_per_doc_logits(artifact_path, module, dl)

    call_count = {"n": 0}
    real_compute = extraction_mod.compute_per_doc_logits_single_doc

    def counting_compute(*args, **kwargs):
        call_count["n"] += 1
        return real_compute(*args, **kwargs)

    extraction_mod.compute_per_doc_logits_single_doc = counting_compute
    try:
        second = persist_single_doc_per_doc_logits(artifact_path, module, dl)
    finally:
        extraction_mod.compute_per_doc_logits_single_doc = real_compute

    assert call_count["n"] == 0
    assert torch.equal(first, second)


def test_compute_per_doc_loo_delta_logits_shape_and_variation(tmp_path) -> None:
    """LOO Δlogit must return (N, K, C) and produce non-zero variance across the K axis.

    Without correct mask construction, leaving each doc out would yield identical
    logits and the K-axis variance would be zero.
    """
    from medrap.extraction import compute_per_doc_loo_delta_logits

    torch.manual_seed(0)
    k_docs = 3
    module = _make_cross_attention_module(k=k_docs)
    batch_a = make_supervised_batch()
    batch_b = make_supervised_batch()
    dl = DataLoader([batch_a, batch_b], batch_size=None)

    delta = compute_per_doc_loo_delta_logits(module, dl)

    n_total = batch_a.code.shape[0] + batch_b.code.shape[0]
    assert delta.shape == (n_total, k_docs, 1)
    assert delta.dtype.is_floating_point
    assert delta.std(dim=1).max().item() > 0.0


def test_compute_per_doc_loo_delta_logits_matches_manual_loo() -> None:
    """For each doc k, Δ[i, k] must equal full_logit[i] - leave_k_out_logit[i]."""
    from medrap.extraction import compute_per_doc_loo_delta_logits

    torch.manual_seed(0)
    k_docs = 3
    module = _make_cross_attention_module(k=k_docs)
    module.eval()
    batch = make_supervised_batch()
    dl = DataLoader([batch], batch_size=None)

    delta = compute_per_doc_loo_delta_logits(module, dl)

    with torch.no_grad():
        encoder_out = module.model.encoder(batch)
        query_out = module.model.query_projector(encoder_out.patient_state)
        retriever_out = module.model.retriever(query_out.query_embeddings)
        retrieval_encoded = module.model.retrieval_encoder(retriever_out)
        rm = retrieval_encoded.retrieval_memory  # (B, R, K, S_doc, D_mem)
        mask = retriever_out.doc_attention_mask  # (B, R, K, S_doc) or None

        # Full forward.
        full_fused = module.model.fusion(
            FusionInput(
                patient_state=encoder_out.patient_state,
                retrieval_memory=rm,
                retrieval_step_ids=query_out.retrieval_step_ids,
                doc_attention_mask=mask,
            )
        )
        full_logit = module.model.head(module.model.pooling(full_fused.fused_state))

        for k in range(k_docs):
            keep = [j for j in range(k_docs) if j != k]
            rm_loo = rm[:, :, keep, :, :]
            mask_loo = mask[:, :, keep, :] if mask is not None else None
            loo_fused = module.model.fusion(
                FusionInput(
                    patient_state=encoder_out.patient_state,
                    retrieval_memory=rm_loo,
                    retrieval_step_ids=query_out.retrieval_step_ids,
                    doc_attention_mask=mask_loo,
                )
            )
            loo_logit = module.model.head(module.model.pooling(loo_fused.fused_state))
            expected = (full_logit - loo_logit).detach().cpu()
            torch.testing.assert_close(delta[:, k, :], expected)


def test_persist_loo_per_doc_logits_creates_key_and_idempotent(tmp_path) -> None:
    """First call adds `per_doc_loo_delta_logits` to the .pt; second call is a cache hit."""
    from medrap import extraction as extraction_mod
    from medrap.extraction import persist_loo_per_doc_logits

    torch.manual_seed(0)
    module = _make_cross_attention_module(k=3)
    batch = make_supervised_batch()
    dl = DataLoader([batch], batch_size=None)
    trainer = lightning.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)
    artifact_path = extract_artifacts(module, dl, trainer, output_dir=tmp_path / "artifacts")

    pre = torch.load(artifact_path, weights_only=True)
    assert "per_doc_loo_delta_logits" not in pre

    first = persist_loo_per_doc_logits(artifact_path, module, dl)

    post = torch.load(artifact_path, weights_only=True)
    assert "per_doc_loo_delta_logits" in post
    assert torch.equal(post["per_doc_loo_delta_logits"], first)
    assert first.shape == (batch.code.shape[0], 3, 1)
    for key in pre:
        assert torch.equal(post[key], pre[key])

    # Second call must hit the cache without re-running the model.
    call_count = {"n": 0}
    real_compute = extraction_mod.compute_per_doc_loo_delta_logits

    def counting_compute(*args, **kwargs):
        call_count["n"] += 1
        return real_compute(*args, **kwargs)

    extraction_mod.compute_per_doc_loo_delta_logits = counting_compute
    try:
        second = persist_loo_per_doc_logits(artifact_path, module, dl)
    finally:
        extraction_mod.compute_per_doc_loo_delta_logits = real_compute

    assert call_count["n"] == 0
    assert torch.equal(first, second)


def test_extract_artifacts_checkpoint_round_trip_preserves_values(tmp_path) -> None:
    """Saving state_dict, re-creating the module, and re-extracting must give identical tensors."""
    torch.manual_seed(42)
    module_a = _make_marginalized_module()
    module_a.eval()
    batch = make_supervised_batch()
    dl = DataLoader([batch], batch_size=None)
    trainer_a = lightning.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)

    path_a = extract_artifacts(module_a, dl, trainer_a, output_dir=tmp_path / "run_a")
    artifacts_a = torch.load(path_a, weights_only=True)

    ckpt_path = tmp_path / "module.ckpt"
    torch.save({"state_dict": module_a.state_dict()}, ckpt_path)

    torch.manual_seed(0)  # intentionally different init to prove the load overrides weights
    module_b = _make_marginalized_module()
    module_b.load_state_dict(torch.load(ckpt_path, weights_only=False)["state_dict"])
    module_b.eval()

    trainer_b = lightning.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)
    path_b = extract_artifacts(module_b, dl, trainer_b, output_dir=tmp_path / "run_b")
    artifacts_b = torch.load(path_b, weights_only=True)

    assert sorted(artifacts_a.keys()) == sorted(artifacts_b.keys())
    for key in artifacts_a:
        assert torch.equal(artifacts_a[key], artifacts_b[key]), (
            f"tensor {key!r} changed across checkpoint round-trip"
        )
