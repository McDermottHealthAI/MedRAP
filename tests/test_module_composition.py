import pytest
import torch
from meds_torchdata import MEDSTorchBatch
from torch import nn

import medrap.retrievers as retrievers_module
from medrap.encoders import MEDSCodeEncoder
from medrap.fusion import ReplaceFusion
from medrap.model import RetrievalAugmentedModel
from medrap.pooling import IdentityPooling
from medrap.query_projection import SequenceMeanQueryProjector
from medrap.retrieval_encoder import (
    LinearProjectionRetrievalEncoder,
    MeanPooledRetrievalEncoder,
    PerDocMeanPooledRetrievalEncoder,
    TokenFeatureRetrievalEncoder,
)
from medrap.retrievers import InMemoryRetriever, load_hf_dataset_retriever
from medrap.types import FusionInput, RetrieverOutput


class TrainableHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def predict(self, pooled_state: torch.Tensor) -> torch.Tensor:
        return self.linear(pooled_state)

    def forward(self, pooled_state: torch.Tensor) -> torch.Tensor:
        return self.predict(pooled_state)


def _example_batch() -> MEDSTorchBatch:
    return MEDSTorchBatch(
        code=torch.LongTensor([[11, 22, 0], [7, 3, 1]]),
        numeric_value=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numeric_value_mask=torch.BoolTensor([[False, False, False], [False, False, False]]),
        time_delta_days=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_trainable_stage_parameters_are_registered_on_model() -> None:
    model = RetrievalAugmentedModel(
        encoder=MEDSCodeEncoder(),
        query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
        retriever=InMemoryRetriever(
            doc_key_embeddings=torch.FloatTensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            doc_tokens=torch.LongTensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
            doc_attention_mask=torch.BoolTensor([[True, True, True, True], [True, True, True, True]]),
        ),
        retrieval_encoder=MeanPooledRetrievalEncoder(vocab_size=8, embedding_dim=2),
        fusion=ReplaceFusion(),
        pooling=IdentityPooling(),
        head=TrainableHead(),
    )

    out = model(_example_batch())

    assert isinstance(out.logits, torch.Tensor)
    assert out.logits.shape == (2, 2)
    assert "head.linear.weight" in model.state_dict()


def test_stage_forward_aliases_named_methods() -> None:
    batch = _example_batch()
    encoder = MEDSCodeEncoder()
    enc_via_method = encoder.encode(batch)
    enc_via_forward = encoder(batch)
    assert torch.equal(enc_via_method.patient_state, enc_via_forward.patient_state)

    projector = SequenceMeanQueryProjector(in_dim=1, out_dim=4)
    q_via_method = projector.project(enc_via_method.patient_state)
    q_via_forward = projector(enc_via_method.patient_state)
    assert torch.equal(q_via_method.query_embeddings, q_via_forward.query_embeddings)
    assert q_via_method.query_embeddings.shape == (2, 1, 4)

    retrieval_out = RetrieverOutput(
        doc_tokens=torch.LongTensor([[1, 2], [3, 4]]),
        doc_attention_mask=torch.LongTensor([[1, 1], [1, 1]]),
    )
    sequence_retrieval_encoder = TokenFeatureRetrievalEncoder(vocab_size=8, embedding_dim=2)
    r_via_method = sequence_retrieval_encoder.encode(retrieval_out)
    r_via_forward = sequence_retrieval_encoder(retrieval_out)
    assert torch.equal(r_via_method.retrieval_memory, r_via_forward.retrieval_memory)
    assert r_via_method.retrieval_memory.shape == (2, 2, 2)
    assert r_via_method.retrieval_memory.dtype == torch.float32

    pooled_retrieval_encoder = MeanPooledRetrievalEncoder(vocab_size=8, embedding_dim=2)
    pooled_via_method = pooled_retrieval_encoder.encode(retrieval_out)
    pooled_via_forward = pooled_retrieval_encoder(retrieval_out)
    assert torch.equal(pooled_via_method.retrieval_memory, pooled_via_forward.retrieval_memory)
    assert pooled_via_method.retrieval_memory.shape == (2, 1, 1, 1, 2)
    assert pooled_via_method.retrieval_memory.dtype == torch.float32

    fusion = ReplaceFusion()
    f_via_method = fusion.fuse(
        FusionInput(
            patient_state=enc_via_method.patient_state,
            retrieval_memory=pooled_via_method.retrieval_memory,
        )
    )
    f_via_forward = fusion(
        FusionInput(
            patient_state=enc_via_method.patient_state,
            retrieval_memory=pooled_via_method.retrieval_memory,
        )
    )
    assert torch.equal(f_via_method.fused_state, f_via_forward.fused_state)

    pooling = IdentityPooling()
    p_via_method = pooling.pool(f_via_method.fused_state)
    p_via_forward = pooling(f_via_method.fused_state)
    assert torch.equal(p_via_method, p_via_forward)


def test_linear_projection_retrieval_encoder() -> None:
    enc = LinearProjectionRetrievalEncoder(in_dim=4, out_dim=2)
    ro = RetrieverOutput(
        doc_tokens=torch.zeros(2, 1, 3, 1, dtype=torch.long),
        doc_attention_mask=torch.ones(2, 1, 3, 1, dtype=torch.bool),
        doc_key_embeddings=torch.randn(2, 1, 3, 4),
    )
    out_method = enc.encode(ro)
    out_forward = enc(ro)
    assert torch.equal(out_method.retrieval_memory, out_forward.retrieval_memory)
    assert out_method.retrieval_memory.shape == (2, 1, 3, 1, 2)
    assert out_method.retrieval_memory.dtype == torch.float32


def test_linear_projection_retrieval_encoder_no_keys_raises() -> None:
    enc = LinearProjectionRetrievalEncoder(in_dim=4, out_dim=2)
    ro = RetrieverOutput(
        doc_tokens=torch.zeros(1, 1, 1, 1, dtype=torch.long),
        doc_attention_mask=torch.ones(1, 1, 1, 1, dtype=torch.bool),
    )
    try:
        enc(ro)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "doc_key_embeddings" in str(e)


def test_linear_projection_preserves_dtype() -> None:
    enc = LinearProjectionRetrievalEncoder(in_dim=4, out_dim=2)
    enc = enc.to(torch.float64)
    ro = RetrieverOutput(
        doc_tokens=torch.zeros(1, 1, 1, 1, dtype=torch.long),
        doc_attention_mask=torch.ones(1, 1, 1, 1, dtype=torch.bool),
        doc_key_embeddings=torch.randn(1, 1, 1, 4, dtype=torch.float64),
    )
    out = enc(ro)
    assert out.retrieval_memory.dtype == torch.float64


def test_per_doc_mean_pooled_retrieval_encoder_shape() -> None:
    enc = PerDocMeanPooledRetrievalEncoder(vocab_size=8, embedding_dim=4)
    ro = RetrieverOutput(
        doc_tokens=torch.LongTensor([[[[1, 2, 0], [3, 0, 0]]]]),
        doc_attention_mask=torch.BoolTensor([[[[True, True, False], [True, False, False]]]]),
    )
    out_method = enc.encode(ro)
    out_forward = enc(ro)
    assert torch.equal(out_method.retrieval_memory, out_forward.retrieval_memory)
    assert out_method.retrieval_memory.shape == (1, 1, 2, 1, 4)
    assert out_method.retrieval_memory.dtype == torch.float32


def test_per_doc_mean_pooled_masking() -> None:
    enc = PerDocMeanPooledRetrievalEncoder(vocab_size=8, embedding_dim=4)
    # Padding token (different id, same mask) should not affect result
    ro_a = RetrieverOutput(
        doc_tokens=torch.LongTensor([[[[1, 0]]]]),
        doc_attention_mask=torch.BoolTensor([[[[True, False]]]]),
    )
    ro_b = RetrieverOutput(
        doc_tokens=torch.LongTensor([[[[1, 7]]]]),
        doc_attention_mask=torch.BoolTensor([[[[True, False]]]]),
    )
    assert torch.equal(enc(ro_a).retrieval_memory, enc(ro_b).retrieval_memory)


def test_per_doc_mean_pooled_all_masked_gives_zero() -> None:
    enc = PerDocMeanPooledRetrievalEncoder(vocab_size=8, embedding_dim=4)
    ro = RetrieverOutput(
        doc_tokens=torch.LongTensor([[[[1, 2]]]]),
        doc_attention_mask=torch.BoolTensor([[[[False, False]]]]),
    )
    assert enc(ro).retrieval_memory.sum().item() == 0.0


def test_in_memory_retriever_returns_query_dependent_payloads() -> None:
    retriever = InMemoryRetriever(
        doc_key_embeddings=torch.FloatTensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        doc_tokens=torch.LongTensor([[10, 11], [20, 21], [30, 31]]),
        doc_attention_mask=torch.BoolTensor([[True, True], [True, True], [True, False]]),
        doc_ids=torch.LongTensor([100, 200, 300]),
        k=2,
    )

    out = retriever.retrieve(torch.FloatTensor([[[-0.1, 1.2]], [[1.0, -0.1]]]))

    assert out.doc_ids is not None
    assert out.doc_ids.tolist() == [[[200, 300]], [[100, 300]]]
    assert out.doc_scores is not None and out.doc_scores.shape == (2, 1, 2)
    assert out.doc_key_embeddings is not None
    assert out.doc_tokens.shape == (2, 1, 2, 2)
    assert out.doc_attention_mask.dtype == torch.bool


def test_load_hf_dataset_retriever_passes_device_and_cache_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeDataset:
        def load_faiss_index(self, index_name, index_path, *, device=None):
            captured["load_faiss_index"] = (index_name, index_path, device)

    class FakeRetriever:
        def __init__(self, **kwargs):
            self._dataset = kwargs["dataset"]
            captured["retriever_kwargs"] = kwargs

    monkeypatch.setattr(retrievers_module, "load_from_disk", lambda path: FakeDataset())
    monkeypatch.setattr(retrievers_module, "HFDatasetRetriever", FakeRetriever)

    retriever = load_hf_dataset_retriever(
        dataset_path="/tmp/retrieval_db",
        index_name="retrieval",
        doc_tokens_column="doc_tokens",
        doc_attention_mask_column="doc_attention_mask",
        k=3,
        device=0,
        ablation_mode="random_docs",
        cache_payloads=True,
        payload_cache_device="cpu",
    )

    assert isinstance(retriever, FakeRetriever)
    assert captured["load_faiss_index"] == (
        "retrieval",
        "/tmp/retrieval_db/retrieval.faiss",
        0,
    )
    assert captured["retriever_kwargs"] == {
        "dataset": retriever._dataset,
        "index_name": "retrieval",
        "doc_tokens_column": "doc_tokens",
        "doc_attention_mask_column": "doc_attention_mask",
        "k": 3,
        "doc_ids_column": None,
        "doc_key_embeddings_column": None,
        "ablation_mode": "random_docs",
        "cache_payloads": True,
        "payload_cache_device": "cpu",
    }


def test_load_hf_dataset_retriever_explains_gpu_faiss_failure(monkeypatch) -> None:
    class FakeDataset:
        def load_faiss_index(self, index_name, index_path, *, device=None):
            raise ImportError("no gpu faiss")

    monkeypatch.setattr(retrievers_module, "load_from_disk", lambda path: FakeDataset())

    with pytest.raises(RuntimeError, match="FAISS GPU retrieval was requested"):
        load_hf_dataset_retriever(
            dataset_path="/tmp/retrieval_db",
            index_name="retrieval",
            doc_tokens_column="doc_tokens",
            doc_attention_mask_column="doc_attention_mask",
            device=0,
        )


def test_load_hf_dataset_retriever_reraises_cpu_faiss_failure(monkeypatch) -> None:
    class FakeDataset:
        def load_faiss_index(self, index_name, index_path, *, device=None):
            assert device is None
            raise FileNotFoundError("missing index")

    monkeypatch.setattr(retrievers_module, "load_from_disk", lambda path: FakeDataset())

    with pytest.raises(FileNotFoundError, match="missing index"):
        load_hf_dataset_retriever(
            dataset_path="/tmp/retrieval_db",
            index_name="retrieval",
            doc_tokens_column="doc_tokens",
            doc_attention_mask_column="doc_attention_mask",
        )
