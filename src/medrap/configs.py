"""Structured config objects and instantiation helpers.

This module provides a minimal hydra-zen based configuration layer for composing the scaffold RAP pipeline
from concrete components.
"""

from dataclasses import dataclass, field
from typing import Any, cast

import lightning
import torch
from datasets import load_dataset, load_from_disk
from hydra.core.config_store import ConfigStore
from hydra_zen import builds, instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from meds_torchdata import MEDSTorchBatch, MEDSTorchDataConfig
from meds_torchdata.extensions.lightning_datamodule import Datamodule as MEDSLightningDatamodule
from meds_torchdata.types import SubsequenceSamplingStrategy
from omegaconf import MISSING
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .model.encoders import MEDSCodeEncoder, TokenEmbeddingEncoder
from .model.fusion import ConcatFusion, ReplaceFusion
from .model.heads import LinearHead
from .model.pooling import IdentityPooling, MaskedMeanPooling
from .model.query_projection import LinearQueryProjector, SequenceMeanQueryProjector
from .model.retrieval_encoder import (
    KeyEmbeddingRetrievalEncoder,
    LinearProjectionRetrievalEncoder,
    MeanPooledRetrievalEncoder,
    PerDocMeanPooledRetrievalEncoder,
    TokenFeatureRetrievalEncoder,
)
from .model.retrievers import InMemoryRetriever, load_hf_dataset_retriever
from .prepare_retrieval.preparation import OrderedFieldDocumentRenderer
from .train.datamodule import SyntheticSupervisedDatamodule
from .train.lightning_module import MedRAPSupervisedLightningModule
from .train.losses import BinaryClassificationLoss, MarginalizedRetrievalSupervisedLoss
from .train.task import (
    BinaryClassificationTask,
    MarginalizedBinaryClassificationTask,
)

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


def meds_torch_batch_config(*, code: Any, boolean_value: Any) -> Any:
    """Return a Hydra-instantiable ``MEDSTorchBatch`` config for binary supervision."""
    return builds_any(
        MEDSTorchBatch,
        code=code,
        numeric_value=float_tensor_config([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numeric_value_mask=bool_tensor_config([[False, False, False], [False, False, False]]),
        time_delta_days=float_tensor_config([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        boolean_value=boolean_value,
        zen_dataclass={"cls_name": "MEDSTorchBatchConfig"},
    )


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
HFDatasetRetrieverConfig = builds_any(
    load_hf_dataset_retriever,
    dataset_path=MISSING,
    index_name="retrieval",
    index_path=None,
    doc_tokens_column="doc_tokens",
    doc_attention_mask_column="doc_attention_mask",
    doc_ids_column="doc_ids",
    doc_key_embeddings_column="doc_key_embeddings",
    k=1,
    device=None,
    ablation_mode="none",
    cache_payloads=False,
    payload_cache_device=None,
    zen_dataclass={"cls_name": "HFDatasetRetrieverConfig"},
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
KeyEmbeddingRetrievalEncoderConfig = builds_any(
    KeyEmbeddingRetrievalEncoder,
    zen_dataclass={"cls_name": "KeyEmbeddingRetrievalEncoderConfig"},
)
LinearProjectionRetrievalEncoderConfig = builds_any(
    LinearProjectionRetrievalEncoder,
    in_dim=1024,
    out_dim=1024,
    zen_dataclass={"cls_name": "LinearProjectionRetrievalEncoderConfig"},
)
PerDocMeanPooledRetrievalEncoderConfig = builds_any(
    PerDocMeanPooledRetrievalEncoder,
    vocab_size=1024,
    embedding_dim=4,
    pretrained_model_name_or_path=None,
    zen_dataclass={"cls_name": "PerDocMeanPooledRetrievalEncoderConfig"},
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
BinaryClassificationTaskConfig = builds_any(
    BinaryClassificationTask,
    zen_dataclass={"cls_name": "BinaryClassificationTaskConfig"},
)
BinaryClassificationLossConfig = builds_any(
    BinaryClassificationLoss,
    zen_dataclass={"cls_name": "BinaryClassificationLossConfig"},
)
MarginalizedBinaryClassificationTaskConfig = builds_any(
    MarginalizedBinaryClassificationTask,
    zen_dataclass={"cls_name": "MarginalizedBinaryClassificationTaskConfig"},
)
MarginalizedRetrievalSupervisedLossConfig = builds_any(
    MarginalizedRetrievalSupervisedLoss,
    zen_dataclass={"cls_name": "MarginalizedRetrievalSupervisedLossConfig"},
)
MedRAPSupervisedLightningModuleConfig = builds_any(
    MedRAPSupervisedLightningModule,
    diagnostics_every_n_steps=50,
    validation_auroc=True,
    validation_auroc_log_per_task=False,
    zen_dataclass={"cls_name": "MedRAPSupervisedLightningModuleConfig"},
)
LoadHFSourceConfig = builds_any(
    load_dataset,
    path=MISSING,
    split=MISSING,
    name=None,
    data_files=None,
    zen_dataclass={"cls_name": "LoadHFSourceConfig"},
)
LoadHFDatasetFromDiskConfig = builds_any(
    load_from_disk,
    dataset_path=MISSING,
    zen_dataclass={"cls_name": "LoadHFDatasetFromDiskConfig"},
)
OrderedFieldDocumentRendererConfig = builds_any(
    OrderedFieldDocumentRenderer,
    fields=MISSING,
    separator="\n",
    include_field_names=False,
    zen_dataclass={"cls_name": "OrderedFieldDocumentRendererConfig"},
)


HFTokenizerConfig = builds_any(
    AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=MISSING,
    zen_dataclass={"cls_name": "HFTokenizerConfig"},
)
SentenceTransformerEmbedderConfig = builds_any(
    SentenceTransformer,
    model_name_or_path=MISSING,
    device="cpu",
    zen_dataclass={"cls_name": "SentenceTransformerEmbedderConfig"},
)


CSVLoggerConfig = builds_any(
    CSVLogger,
    save_dir="${default_root_dir}/loggers",
    name="csv",
    zen_dataclass={"cls_name": "CSVLoggerConfig"},
)
DefaultModelCheckpointConfig = builds_any(
    ModelCheckpoint,
    dirpath="${default_root_dir}/checkpoints",
    filename="epoch={epoch}-step={step}",
    monitor="val/loss",
    mode="min",
    save_last=True,
    auto_insert_metric_name=False,
    zen_dataclass={"cls_name": "DefaultModelCheckpointConfig"},
)
LightningDemoTrainerConfig = builds_any(
    lightning.Trainer,
    default_root_dir=".",
    max_epochs=1,
    accelerator="cpu",
    devices=1,
    logger=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=False,
    log_every_n_steps=1,
    zen_dataclass={"cls_name": "LightningDemoTrainerConfig"},
)
LightningDefaultTrainerConfig = builds_any(
    lightning.Trainer,
    default_root_dir=".",
    max_epochs=10,
    accelerator="auto",
    devices=1,
    logger=CSVLoggerConfig(),
    callbacks=[DefaultModelCheckpointConfig()],
    enable_checkpointing=True,
    enable_model_summary=False,
    enable_progress_bar=False,
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    log_every_n_steps=10,
    zen_dataclass={"cls_name": "LightningDefaultTrainerConfig"},
)
LightningEvalTrainerConfig = builds_any(
    lightning.Trainer,
    default_root_dir=".",
    max_epochs=1,
    accelerator="cpu",
    devices=1,
    logger=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=False,
    log_every_n_steps=1,
    zen_dataclass={"cls_name": "LightningEvalTrainerConfig"},
)
MEDSTorchDataConfigConfig = builds_any(
    MEDSTorchDataConfig,
    tensorized_cohort_dir=MISSING,
    max_seq_len=MISSING,
    task_labels_dir=MISSING,
    seq_sampling_strategy=SubsequenceSamplingStrategy.TO_END,
    zen_dataclass={"cls_name": "MEDSTorchDataConfigConfig"},
)
MEDSTrainingDatamoduleConfig = builds_any(
    MEDSLightningDatamodule,
    config=MEDSTorchDataConfigConfig(),
    batch_size=32,
    num_workers=None,
    pin_memory=None,
    zen_dataclass={"cls_name": "MEDSTrainingDatamoduleConfig"},
)
SyntheticSupervisedDatamoduleConfig = builds_any(
    SyntheticSupervisedDatamodule,
    train_batch=meds_torch_batch_config(
        code=long_tensor_config([[1, 2, 3], [3, 2, 1]]),
        boolean_value=bool_tensor_config([True, False]),
    ),
    val_batch=None,
    test_batch=None,
    train_repeat=2,
    val_repeat=1,
    test_repeat=1,
    zen_dataclass={"cls_name": "SyntheticSupervisedDatamoduleConfig"},
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
    marginalized_retrieval: bool = False
    marginalized_score_similarity: str = "dot"
    marginalized_output_mode: str = "categorical"  # "categorical" | "binary"


@dataclass
class RAPAppConfig(PipelineConfig):
    """Top-level app config for CLI/Hydra composition."""

    @classmethod
    def add_to_config_store(cls, group: str | None = None) -> None:
        """Register this config in Hydra's ConfigStore.

        This follows the standard ``ConfigStore.store`` pattern used in
        MEDS ecosystem repos.

        Examples:
            >>> RAPAppConfig.add_to_config_store(group="medrap_doctest")
            >>> "RAPAppConfig.yaml" in ConfigStore.instance().repo["medrap_doctest"]
            True
        """
        cs = ConfigStore.instance()
        cs.store(name=cls.__name__, group=group, node=cls)


@dataclass
class TrainingConfig:
    """Minimal training config layer on top of the plain RAP model config."""

    module: ComponentConfig = field(default_factory=MedRAPSupervisedLightningModuleConfig)
    task: ComponentConfig = field(default_factory=BinaryClassificationTaskConfig)
    loss: ComponentConfig = field(default_factory=BinaryClassificationLossConfig)
    trainer: ComponentConfig = field(default_factory=LightningDefaultTrainerConfig)
    datamodule: ComponentConfig = field(default_factory=MEDSTrainingDatamoduleConfig)


@dataclass
class RAPTrainConfig(PipelineConfig):
    """Top-level training config that preserves ``PipelineConfig`` as model composition."""

    head: ComponentConfig = field(default_factory=lambda: LinearHeadConfig(out_dim=1))
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = MISSING
    do_resume: bool = False
    do_overwrite: bool = False


@dataclass
class RAPEvalConfig(PipelineConfig):
    """Top-level eval config mirroring train execution fields."""

    head: ComponentConfig = field(default_factory=lambda: LinearHeadConfig(out_dim=1))
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(trainer=LightningEvalTrainerConfig())
    )
    output_dir: str = MISSING
    checkpoint_path: str = MISSING
    eval_mode: str = "validate"


@dataclass
class RAPRetrieveConfig(PipelineConfig):
    """Top-level retrieve config mirroring eval execution fields."""

    head: ComponentConfig = field(default_factory=lambda: LinearHeadConfig(out_dim=1))
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(trainer=LightningEvalTrainerConfig())
    )
    output_dir: str = MISSING
    checkpoint_path: str = MISSING
    index_dataframe_dir: str = MISSING
    splits: tuple[str, ...] = ("train", "tuning", "held_out")
    batch_size: int = 32


@dataclass
class RetrievalDatasetIndexConfig:
    """Configuration for offline retrieval artifact columns and FAISS index."""

    doc_text_column: str = "doc_text"
    doc_tokens_column: str = "doc_tokens"
    doc_attention_mask_column: str = "doc_attention_mask"
    doc_key_embeddings_column: str = "doc_key_embeddings"
    doc_ids_column: str = "doc_ids"
    source_id_column: str | None = None
    index_name: str = "retrieval"
    max_length: int = 512
    tokenization_batch_size: int = 256
    embedding_batch_size: int = 256
    encode_batch_size: int | None = None
    string_factory: str | None = None


@dataclass
class RetrievalDatasetOutputConfig:
    """Configuration for saved retrieval artifact location."""

    output_dir: str = MISSING


@dataclass
class PrepareRetrievalDatasetConfig:
    """Configuration container for preparing a retrieval dataset artifact."""

    source: ComponentConfig = field(default_factory=LoadHFSourceConfig)
    document: ComponentConfig = field(default_factory=OrderedFieldDocumentRendererConfig)
    tokenizer: ComponentConfig = field(default_factory=HFTokenizerConfig)
    embedder: ComponentConfig = field(default_factory=SentenceTransformerEmbedderConfig)
    index: RetrievalDatasetIndexConfig = field(default_factory=RetrievalDatasetIndexConfig)
    output: RetrievalDatasetOutputConfig = field(default_factory=RetrievalDatasetOutputConfig)
    num_docs: int | None = None
    num_docs_seed: int = 42


@dataclass
class PrepareRetrievalDatasetAppConfig:
    """Top-level app config for retrieval dataset preparation."""

    prep: PrepareRetrievalDatasetConfig = field(default_factory=PrepareRetrievalDatasetConfig)
    do_overwrite: bool = False


@dataclass
class PreprocessDatasetAppConfig:
    """Top-level app config for the medrap-preprocess pipeline.

    Stages:
        1. MEDS-transforms (add time features, filter rare codes/sparse subjects,
           quantize numeric values) — skipped when ``tensorized_dir`` is set.
        2. MTD_preprocess (tokenize + tensorize) — skipped when ``tensorized_dir`` is set.
        3. generate_tasks (sample task codes, create binary labels).
    """

    meds_data_dir: str = MISSING
    output_dir: str = MISSING
    tensorized_dir: str | None = None
    min_subjects_per_code: int = 100
    min_events_per_subject: int = 10
    num_tasks: int = 25
    horizon_days: float = 7.0
    min_history_days: float = 1.0
    seed: int = 42
    code_selection: str = "random"  # "random" | "most_frequent"
    duration_distribution: str = "fixed"  # "fixed" | "uniform" | "log-uniform"
    min_duration_days: float | None = None
    max_duration_days: float | None = None
    do_overwrite: bool = False
