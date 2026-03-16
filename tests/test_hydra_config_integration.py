import torch
from datasets import Dataset
from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from meds_torchdata import MEDSTorchBatch
from torch import nn

from medrap.configs import RAPAppConfig, instantiate_training_module
from medrap.lightning_module import MedRAPSupervisedLightningModule
from medrap.model import RetrievalAugmentedModel
from medrap.preparation import OrderedFieldDocumentRenderer, prepare_retrieval_dataset
from medrap.retrievers import HFDatasetRetriever, InMemoryRetriever
from medrap.runtime import build_model_from_cfg
from medrap.task import SupervisedTask


def _example_batch() -> MEDSTorchBatch:
    return MEDSTorchBatch(
        code=torch.LongTensor([[101, 7, 0], [42, 3, 0]]),
        numeric_value=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numeric_value_mask=torch.BoolTensor([[False, False, False], [False, False, False]]),
        time_delta_days=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_train_config_composes_and_instantiates_model() -> None:
    with initialize_config_module(version_base=None, config_module="medrap.conf"):
        cfg = compose(config_name="_train")

    model = build_model_from_cfg(cfg)

    assert isinstance(model, RetrievalAugmentedModel)
    assert isinstance(model.retriever, InMemoryRetriever)
    out = model.forward(_example_batch())
    assert out.logits.shape == (2, 1)
    assert out.logits.dtype == torch.float32


def test_train_config_composes_training_layer() -> None:
    with initialize_config_module(version_base=None, config_module="medrap.conf"):
        cfg = compose(config_name="_train")

    lightning_module = instantiate_training_module(cfg)

    assert isinstance(lightning_module, MedRAPSupervisedLightningModule)
    assert isinstance(lightning_module.model, RetrievalAugmentedModel)
    assert isinstance(lightning_module.task, nn.Module)
    assert lightning_module.loss_fn.__class__.__name__ == "BinaryClassificationLoss"
    assert cfg.training.task.output_dim == 1
    assert cfg.head.out_dim == cfg.training.task.output_dim


def test_eval_config_composes_training_layer() -> None:
    with initialize_config_module(version_base=None, config_module="medrap.conf"):
        cfg = compose(config_name="_eval")

    lightning_module = instantiate_training_module(cfg)

    assert isinstance(lightning_module, MedRAPSupervisedLightningModule)
    assert isinstance(lightning_module.model, RetrievalAugmentedModel)
    assert isinstance(lightning_module.task, nn.Module)
    assert lightning_module.loss_fn.__class__.__name__ == "BinaryClassificationLoss"
    assert cfg.head.out_dim == cfg.training.task.output_dim


def test_app_config_registers_with_hydra_config_store() -> None:
    RAPAppConfig.add_to_config_store(group="medrap")
    cs = ConfigStore.instance()

    assert "medrap" in cs.repo
    assert "RAPAppConfig.yaml" in cs.repo["medrap"]


def test_supervised_task_is_not_exported_from_package_root() -> None:
    import medrap

    assert not hasattr(medrap, "SupervisedTask")
    assert not hasattr(medrap, "SupervisedLoss")
    assert SupervisedTask.__name__ == "SupervisedTask"


class _Tokenizer:
    def __call__(self, texts, *, truncation, padding, max_length):
        return {
            "input_ids": [[idx + 1] * max_length for idx, _ in enumerate(texts)],
            "attention_mask": [[1] * max_length for _ in texts],
        }


class _Embedder:
    def encode(self, texts, *, batch_size, convert_to_numpy, show_progress_bar):
        return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]][: len(texts)]


def test_prepare_retrieval_dataset_config_composes(tmp_path) -> None:
    with initialize_config_module(version_base=None, config_module="medrap.conf"):
        cfg = compose(
            config_name="_prepare_retrieval_dataset",
            overrides=[
                "prep/source=load_from_disk",
                f"prep.source.dataset_path={tmp_path}/source",
                "prep.document.fields=[text]",
                "prep.tokenizer.model_name=stub-tokenizer",
                "prep.embedder.model_name=stub-embedder",
                f"prep.output.output_dir={tmp_path}/prepared",
            ],
        )

    assert cfg.prep.source.dataset_path == f"{tmp_path}/source"
    assert cfg.prep.document.fields == ["text"]
    assert cfg.prep.index.index_name == "retrieval"


def test_eval_config_supports_saved_hf_dataset_retriever(tmp_path) -> None:
    artifact_dir = prepare_retrieval_dataset(
        dataset=Dataset.from_dict({"text": ["alpha", "beta"]}),
        renderer=OrderedFieldDocumentRenderer(fields=["text"]),
        tokenizer=_Tokenizer(),
        embedder=_Embedder(),
        output_dir=tmp_path / "prepared",
        max_length=4,
    )

    with initialize_config_module(version_base=None, config_module="medrap.conf"):
        cfg = compose(
            config_name="_eval",
            overrides=[
                "retriever=hf_dataset",
                f"retriever.dataset_path={artifact_dir}",
            ],
        )

    model = build_model_from_cfg(cfg)

    assert isinstance(model, RetrievalAugmentedModel)
    assert isinstance(model.retriever, HFDatasetRetriever)
    out = model.forward(_example_batch())
    assert out.logits.shape == (2, 1)
    assert out.logits.dtype == torch.float32
