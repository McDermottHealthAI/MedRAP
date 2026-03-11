"""Structured config objects and instantiation helpers.

This module provides a minimal hydra-zen based configuration layer for composing the scaffold RAP pipeline
from concrete components.
"""

from dataclasses import dataclass, field
from typing import Any, cast

import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import builds, instantiate

from .encoders import MEDSCodeEncoder, TabularEncoder, TokenEmbeddingEncoder
from .fusion import ConcatFusion, ReplaceFusion
from .heads import LinearHead
from .model import RetrievalAugmentedModel
from .pooling import IdentityPooling, MaskedMeanPooling
from .query_projection import LinearQueryProjector, SequenceMeanQueryProjector
from .retrieval_encoder import MeanPooledRetrievalEncoder, TokenFeatureRetrievalEncoder
from .retrievers import InMemoryRetriever

ComponentConfig = Any
builds_any = cast("Any", builds)
instantiate_any = cast("Any", instantiate)


def long_tensor_config(values: Any) -> Any:
    """Return a Hydra-instantiable ``torch.LongTensor`` config."""
    return builds_any(torch.LongTensor, values, populate_full_signature=False)


def bool_tensor_config(values: Any) -> Any:
    """Return a Hydra-instantiable ``torch.BoolTensor`` config."""
    return builds_any(torch.BoolTensor, values, populate_full_signature=False)


def float_tensor_config(values: Any) -> Any:
    """Return a Hydra-instantiable ``torch.FloatTensor`` config."""
    return builds_any(torch.FloatTensor, values, populate_full_signature=False)


MEDSCodeEncoderConfig = builds_any(
    MEDSCodeEncoder,
    zen_dataclass={"cls_name": "MEDSCodeEncoderConfig"},
)
TokenEmbeddingEncoderConfig = builds_any(
    TokenEmbeddingEncoder,
    vocab_size=1024,
    embedding_dim=4,
    zen_dataclass={"cls_name": "TokenEmbeddingEncoderConfig"},
)
TabularEncoderConfig = builds_any(
    TabularEncoder,
    vocab_size=1024,
    embedding_dim=4,
    zen_dataclass={"cls_name": "TabularEncoderConfig"},
)
LinearQueryProjectorConfig = builds_any(
    LinearQueryProjector,
    in_dim=4,
    out_dim=4,
    zen_dataclass={"cls_name": "LinearQueryProjectorConfig"},
)
SequenceMeanQueryProjectorConfig = builds_any(
    SequenceMeanQueryProjector,
    in_dim=1,
    out_dim=4,
    zen_dataclass={"cls_name": "SequenceMeanQueryProjectorConfig"},
)
InMemoryRetrieverConfig = builds_any(
    InMemoryRetriever,
    populate_full_signature=True,
    zen_dataclass={"cls_name": "InMemoryRetrieverConfig"},
)
DemoInMemoryRetrieverConfig = builds_any(
    InMemoryRetriever,
    populate_full_signature=True,
    doc_key_embeddings=float_tensor_config(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    ),
    doc_tokens=long_tensor_config([[1, 2], [3, 4]]),
    doc_attention_mask=bool_tensor_config([[True, True], [True, True]]),
    zen_dataclass={"cls_name": "DemoInMemoryRetrieverConfig"},
)
TokenFeatureRetrievalEncoderConfig = builds_any(
    TokenFeatureRetrievalEncoder,
    vocab_size=1024,
    embedding_dim=4,
    zen_dataclass={"cls_name": "TokenFeatureRetrievalEncoderConfig"},
)
MeanPooledRetrievalEncoderConfig = builds_any(
    MeanPooledRetrievalEncoder,
    vocab_size=1024,
    embedding_dim=4,
    zen_dataclass={"cls_name": "MeanPooledRetrievalEncoderConfig"},
)
ReplaceFusionConfig = builds_any(
    ReplaceFusion,
    zen_dataclass={"cls_name": "ReplaceFusionConfig"},
)
ConcatFusionConfig = builds_any(
    ConcatFusion,
    zen_dataclass={"cls_name": "ConcatFusionConfig"},
)
IdentityPoolingConfig = builds_any(
    IdentityPooling,
    zen_dataclass={"cls_name": "IdentityPoolingConfig"},
)
MaskedMeanPoolingConfig = builds_any(
    MaskedMeanPooling,
    zen_dataclass={"cls_name": "MaskedMeanPoolingConfig"},
)
LinearHeadConfig = builds_any(
    LinearHead,
    in_dim=4,
    out_dim=2,
    zen_dataclass={"cls_name": "LinearHeadConfig"},
)


@dataclass
class PipelineConfig:
    """Configuration container for composing ``RetrievalAugmentedModel``."""

    # ``object`` keeps Hydra/OmegaConf structured-config compatibility while still
    # allowing stage-specific hydra-zen config objects.
    encoder: ComponentConfig = field(default_factory=MEDSCodeEncoderConfig)
    query_projector: ComponentConfig = field(default_factory=SequenceMeanQueryProjectorConfig)
    retriever: ComponentConfig = field(default_factory=DemoInMemoryRetrieverConfig)
    retrieval_encoder: ComponentConfig = field(default_factory=MeanPooledRetrievalEncoderConfig)
    fusion: ComponentConfig = field(default_factory=ReplaceFusionConfig)
    pooling: ComponentConfig = field(default_factory=IdentityPoolingConfig)
    head: ComponentConfig = field(default_factory=LinearHeadConfig)


@dataclass
class RAPAppConfig(PipelineConfig):
    """Top-level app config for CLI/Hydra composition."""

    @classmethod
    def add_to_config_store(cls, group: str | None = None) -> None:
        """Register this config in Hydra's ConfigStore.

        This follows the standard ``ConfigStore.store`` pattern used in
        MEDS ecosystem repos.
        """
        cs = ConfigStore.instance()
        cs.store(name=cls.__name__, group=group, node=cls)


def default_pipeline_config() -> PipelineConfig:
    """Return a default, fully-instantiable pipeline config."""
    return PipelineConfig()


def instantiate_model(config: Any) -> RetrievalAugmentedModel:
    """Instantiate a ``RetrievalAugmentedModel`` from structured config."""
    return RetrievalAugmentedModel(
        encoder=instantiate_any(config.encoder),
        query_projector=instantiate_any(config.query_projector),
        retriever=instantiate_any(config.retriever),
        retrieval_encoder=instantiate_any(config.retrieval_encoder),
        fusion=instantiate_any(config.fusion),
        pooling=instantiate_any(config.pooling),
        head=instantiate_any(config.head),
    )
