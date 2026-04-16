import lightning
import torch
from torch.utils.data import DataLoader

from conftest import make_supervised_batch
from medrap.encoders import MEDSCodeEncoder
from medrap.extraction import collate_prediction_batches, extract_artifacts
from medrap.fusion import ReplaceFusion
from medrap.heads import LinearHead
from medrap.lightning_module import MedRAPSupervisedLightningModule
from medrap.model import RetrievalAugmentedModel
from medrap.pooling import IdentityPooling
from medrap.query_projection import SequenceMeanQueryProjector
from medrap.retrieval_encoder import MeanPooledRetrievalEncoder
from medrap.retrievers import InMemoryRetriever


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
