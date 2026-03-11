import torch
from meds_torchdata import MEDSTorchBatch
from torch import nn

from medrap.encoders import MEDSCodeEncoder
from medrap.fusion import ReplaceFusion
from medrap.model import RetrievalAugmentedModel
from medrap.pooling import IdentityPooling
from medrap.query_projection import SequenceMeanQueryProjector
from medrap.retrieval_encoder import MeanPooledRetrievalEncoder, TokenFeatureRetrievalEncoder
from medrap.retrievers import InMemoryRetriever
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
    assert pooled_via_method.retrieval_memory.shape == (2, 2)
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
