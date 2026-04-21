import lightning
import pytest
import torch
from torch.utils.data import DataLoader

from conftest import make_supervised_batch
from medrap.encoders import MEDSCodeEncoder
from medrap.extraction import collate_prediction_batches, extract_artifacts
from medrap.fusion import ReplaceFusion
from medrap.heads import LinearHead
from medrap.lightning_module import MedRAPSupervisedLightningModule
from medrap.losses import MarginalizedRetrievalSupervisedLoss
from medrap.model import RetrievalAugmentedModel
from medrap.pooling import IdentityPooling
from medrap.query_projection import SequenceMeanQueryProjector
from medrap.retrieval_encoder import KeyEmbeddingRetrievalEncoder, MeanPooledRetrievalEncoder
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
