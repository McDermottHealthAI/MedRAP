import torch
from meds_torchdata import MEDSTorchBatch

from medrap.configs import (
    ConcatFusionConfig,
    LinearHeadConfig,
    LinearQueryProjectorConfig,
    MaskedMeanPoolingConfig,
    PipelineConfig,
    TokenEmbeddingEncoderConfig,
    TopKPayloadRetrieverConfig,
    bool_tensor_config,
    default_pipeline_config,
    float_tensor_config,
    instantiate_model,
    long_tensor_config,
)
from medrap.encoders import MEDSCodeEncoder, TokenEmbeddingEncoder
from medrap.fusion import ConcatFusion, ReplaceFusion
from medrap.heads import LinearHead
from medrap.model import RetrievalAugmentedModel
from medrap.pooling import IdentityPooling, MaskedMeanPooling
from medrap.query_projection import LinearQueryProjector, SequenceMeanQueryProjector
from medrap.retrieval_encoder import MeanPooledRetrievalEncoder
from medrap.retrievers import TopKPayloadRetriever


def _example_batch() -> MEDSTorchBatch:
    return MEDSTorchBatch(
        code=torch.LongTensor([[101, 7, 0], [42, 3, 0]]),
        numeric_value=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numeric_value_mask=torch.BoolTensor([[False, False, False], [False, False, False]]),
        time_delta_days=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_default_pipeline_config_instantiates_default_components() -> None:
    cfg = default_pipeline_config()

    model = instantiate_model(cfg)

    assert isinstance(model, RetrievalAugmentedModel)
    assert isinstance(model.encoder, MEDSCodeEncoder)
    assert isinstance(model.query_projector, SequenceMeanQueryProjector)
    assert isinstance(model.retriever, TopKPayloadRetriever)
    assert isinstance(model.retrieval_encoder, MeanPooledRetrievalEncoder)
    assert isinstance(model.fusion, ReplaceFusion)
    assert isinstance(model.pooling, IdentityPooling)
    assert isinstance(model.head, LinearHead)


def test_pipeline_config_allows_overriding_retriever_values() -> None:
    cfg = PipelineConfig(
        retriever=TopKPayloadRetrieverConfig(
            doc_key_embeddings=float_tensor_config([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            doc_tokens=long_tensor_config([[9, 8], [7, 6]]),
            doc_attention_mask=bool_tensor_config([[True, True], [True, True]]),
        )
    )
    model = instantiate_model(cfg)

    out = model.forward(_example_batch())

    assert out.logits.shape == (2, 2)
    assert out.logits.dtype == torch.float32


def test_pipeline_config_allows_meaningful_module_overrides() -> None:
    cfg = PipelineConfig(
        encoder=TokenEmbeddingEncoderConfig(vocab_size=32, embedding_dim=3),
        query_projector=LinearQueryProjectorConfig(in_dim=3, out_dim=2),
        retriever=TopKPayloadRetrieverConfig(
            doc_key_embeddings=float_tensor_config([[1.0, 0.0], [0.0, 1.0]]),
            doc_tokens=long_tensor_config([[1, 2, 0], [3, 4, 0]]),
            doc_attention_mask=bool_tensor_config([[True, True, False], [True, True, False]]),
        ),
        fusion=ConcatFusionConfig(),
        pooling=MaskedMeanPoolingConfig(),
        head=LinearHeadConfig(in_dim=5, out_dim=2),
    )
    model = instantiate_model(cfg)

    assert isinstance(model.encoder, TokenEmbeddingEncoder)
    assert isinstance(model.query_projector, LinearQueryProjector)
    assert isinstance(model.fusion, ConcatFusion)
    assert isinstance(model.pooling, MaskedMeanPooling)
    assert isinstance(model.head, LinearHead)
