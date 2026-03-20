"""Structured config objects and instantiation helpers.

This module provides a minimal hydra-zen based configuration layer for composing the scaffold RAP pipeline
from concrete components.
"""

from dataclasses import dataclass, field
from typing import Any, cast

import lightning
import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import builds, instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from meds_torchdata import MEDSTorchBatch, MEDSTorchDataConfig
from meds_torchdata.extensions.lightning_datamodule import Datamodule as MEDSLightningDatamodule
from meds_torchdata.types import SubsequenceSamplingStrategy
from omegaconf import MISSING

from .datamodule import SyntheticSupervisedDatamodule
from .encoders import MEDSCodeEncoder, TabularEncoder, TokenEmbeddingEncoder
from .fusion import ConcatFusion, ReplaceFusion
from .heads import LinearHead
from .lightning_module import MedRAPSupervisedLightningModule
from .model import RetrievalAugmentedModel
from .pooling import IdentityPooling, MaskedMeanPooling
from .query_projection import LinearQueryProjector, SequenceMeanQueryProjector
from .retrieval_encoder import MeanPooledRetrievalEncoder, TokenFeatureRetrievalEncoder
from .retrievers import InMemoryRetriever
from .task import BinaryClassificationLoss, BinaryClassificationTask

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
BinaryClassificationTaskConfig = builds_any(
    BinaryClassificationTask,
    zen_dataclass={"cls_name": "BinaryClassificationTaskConfig"},
)
BinaryClassificationLossConfig = builds_any(
    BinaryClassificationLoss,
    zen_dataclass={"cls_name": "BinaryClassificationLossConfig"},
)
MedRAPSupervisedLightningModuleConfig = builds_any(
    MedRAPSupervisedLightningModule,
    zen_dataclass={"cls_name": "MedRAPSupervisedLightningModuleConfig"},
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


def instantiate_model(config: Any) -> RetrievalAugmentedModel:
    """Instantiate a ``RetrievalAugmentedModel`` from structured config.

    Examples:
        >>> model = instantiate_model(PipelineConfig())
        >>> (
        ...     model.encoder.__class__.__name__,
        ...     model.query_projector.__class__.__name__,
        ...     model.retriever.__class__.__name__,
        ...     model.retrieval_encoder.__class__.__name__,
        ...     model.fusion.__class__.__name__,
        ...     model.pooling.__class__.__name__,
        ...     model.head.__class__.__name__,
        ... )
        >>> names == (
        ...     "MEDSCodeEncoder",
        ...     "SequenceMeanQueryProjector",
        ...     "InMemoryRetriever",
        ...     "MeanPooledRetrievalEncoder",
        ...     "ReplaceFusion",
        ...     "IdentityPooling",
        ...     "LinearHead",
        ... )
        True
        >>> model = instantiate_model(
        ...     PipelineConfig(
        ...         encoder=TokenEmbeddingEncoderConfig(vocab_size=32, embedding_dim=3),
        ...         query_projector=LinearQueryProjectorConfig(in_dim=3, out_dim=2),
        ...         retriever=InMemoryRetrieverConfig(
        ...             doc_key_embeddings=float_tensor_config([[1.0, 0.0], [0.0, 1.0]]),
        ...             doc_tokens=long_tensor_config([[9, 8, 0], [7, 6, 0]]),
        ...             doc_attention_mask=bool_tensor_config([[True, True, False], [True, True, False]]),
        ...         ),
        ...         fusion=ConcatFusionConfig(),
        ...         pooling=MaskedMeanPoolingConfig(),
        ...         head=LinearHeadConfig(in_dim=5, out_dim=2),
        ...     )
        ... )
        >>> (
        ...     model.encoder.__class__.__name__,
        ...     model.query_projector.__class__.__name__,
        ...     model.fusion.__class__.__name__,
        ...     model.pooling.__class__.__name__,
        ...     model.head.__class__.__name__,
        ... )
        ('TokenEmbeddingEncoder', 'LinearQueryProjector', 'ConcatFusion', 'MaskedMeanPooling', 'LinearHead')
        >>> model.head.linear.in_features
        5
    """
    return RetrievalAugmentedModel(
        encoder=instantiate_any(config.encoder),
        query_projector=instantiate_any(config.query_projector),
        retriever=instantiate_any(config.retriever),
        retrieval_encoder=instantiate_any(config.retrieval_encoder),
        fusion=instantiate_any(config.fusion),
        pooling=instantiate_any(config.pooling),
        head=instantiate_any(config.head),
    )


def instantiate_training_module(config: RAPTrainConfig) -> MedRAPSupervisedLightningModule:
    """Instantiate the configured training wrapper around the plain RAP model.

    Args:
        config: Training config containing the plain RAP model composition under the
            top-level pipeline fields and the supervised wrapper/task under
            ``config.training``.

    Returns:
        MedRAPSupervisedLightningModule: Lightning wrapper whose plain model returns
        logits shaped ``(B, config.training.task.output_dim)`` for a batch of size
        ``B``.

    Examples:
        >>> module = instantiate_training_module(RAPTrainConfig(output_dir="outputs/demo"))
        >>> module.__class__.__name__
        'MedRAPSupervisedLightningModule'
        >>> module.task.output_dim
        1
        >>> module.loss_fn.__class__.__name__
        'BinaryClassificationLoss'
        >>> batch = MEDSTorchBatch(
        ...     code=torch.LongTensor([[101, 7, 0], [42, 3, 0]]),
        ...     numeric_value=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ...     numeric_value_mask=torch.BoolTensor([[False, False, False], [False, False, False]]),
        ...     time_delta_days=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ...     boolean_value=torch.BoolTensor([True, False]),
        ... )
        >>> out = module.model.forward(batch)
        >>> tuple(out.logits.shape)
        (2, 1)
        >>> targets = module.task.extract_targets(batch)
        >>> module.loss_fn(out, targets).ndim
        0
    """
    plain_model = instantiate_model(config)
    task = instantiate_any(config.training.task)
    loss_fn = instantiate_any(config.training.loss)
    return instantiate_any(config.training.module, model=plain_model, task=task, loss_fn=loss_fn)


def instantiate_datamodule(config: RAPTrainConfig | RAPEvalConfig) -> lightning.LightningDataModule:
    """Instantiate the configured training datamodule.

    Args:
        config: Execution config containing datamodule settings under
            ``config.training.datamodule``.

    Returns:
        lightning.LightningDataModule: Configured datamodule yielding ``MEDSTorchBatch``.

    Examples:
        >>> datamodule = instantiate_datamodule(
        ...     RAPTrainConfig(
        ...         output_dir="outputs/demo",
        ...         training=TrainingConfig(datamodule=SyntheticSupervisedDatamoduleConfig()),
        ...     )
        ... )
        >>> datamodule.__class__.__name__
        'SyntheticSupervisedDatamodule'
    """
    return instantiate_any(config.training.datamodule)
