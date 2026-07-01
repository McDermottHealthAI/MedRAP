import torch
from datasets import Dataset
from hydra import compose, initialize_config_module
from hydra_zen import instantiate
from meds_torchdata import MEDSTorchBatch
from torch import nn

from medrap.model.model import RetrievalAugmentedModel
from medrap.model.retrievers import HFDatasetRetriever
from medrap.prepare_retrieval.preparation import OrderedFieldDocumentRenderer, prepare_retrieval_dataset
from medrap.train.factory import instantiate_training_module
from medrap.train.lightning_module import MedRAPSupervisedLightningModule
from medrap.train.task import SupervisedTask


def _example_batch() -> MEDSTorchBatch:
    return MEDSTorchBatch(
        code=torch.LongTensor([[101, 7, 0], [42, 3, 0]]),
        numeric_value=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numeric_value_mask=torch.BoolTensor([[False, False, False], [False, False, False]]),
        time_delta_days=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_train_config_composes_training_layer() -> None:
    with initialize_config_module(version_base=None, config_module="medrap.conf"):
        cfg = compose(
            config_name="_train",
            overrides=[
                "training/datamodule=synthetic",
                "output_dir=outputs/medrap-test",
            ],
        )

    lightning_module = instantiate_training_module(cfg)
    datamodule = instantiate(cfg.training.datamodule)

    assert isinstance(lightning_module, MedRAPSupervisedLightningModule)
    assert isinstance(lightning_module.model, RetrievalAugmentedModel)
    assert isinstance(lightning_module.task, nn.Module)
    assert lightning_module.loss_fn.__class__.__name__ == "BinaryClassificationLoss"
    assert datamodule.__class__.__name__ == "SyntheticSupervisedDatamodule"
    assert cfg.training.task.output_dim == 1
    assert cfg.head.out_dim == cfg.training.task.output_dim
    assert cfg.training.trainer.gradient_clip_val == 1.0
    assert cfg.training.trainer.gradient_clip_algorithm == "norm"


def test_eval_config_composes_training_layer() -> None:
    with initialize_config_module(version_base=None, config_module="medrap.conf"):
        cfg = compose(
            config_name="_eval",
            overrides=[
                "training/datamodule=synthetic",
                "output_dir=outputs/medrap-eval",
                "checkpoint_path=outputs/medrap-eval/checkpoints/last.ckpt",
            ],
        )

    lightning_module = instantiate_training_module(cfg)
    datamodule = instantiate(cfg.training.datamodule)

    assert isinstance(lightning_module, MedRAPSupervisedLightningModule)
    assert isinstance(lightning_module.model, RetrievalAugmentedModel)
    assert isinstance(lightning_module.task, nn.Module)
    assert lightning_module.loss_fn.__class__.__name__ == "BinaryClassificationLoss"
    assert datamodule.__class__.__name__ == "SyntheticSupervisedDatamodule"
    assert cfg.head.out_dim == cfg.training.task.output_dim


def test_train_config_supports_meds_datamodule_overrides(tmp_path) -> None:
    cohort_dir = tmp_path / "tensorized"
    labels_dir = tmp_path / "labels"
    cohort_dir.mkdir()
    labels_dir.mkdir()

    with initialize_config_module(version_base=None, config_module="medrap.conf"):
        cfg = compose(
            config_name="_train",
            overrides=[
                "training/datamodule=meds",
                f"output_dir={tmp_path / 'outputs'}",
                f"training.datamodule.config.tensorized_cohort_dir={cohort_dir}",
                "training.datamodule.config.max_seq_len=8",
                f"training.datamodule.config.task_labels_dir={labels_dir}",
            ],
        )

    datamodule = instantiate(cfg.training.datamodule)

    assert datamodule.__class__.__name__ == "Datamodule"


def test_supervised_task_is_not_exported_from_package_root() -> None:
    import medrap

    assert not hasattr(medrap, "SupervisedTask")
    assert not hasattr(medrap, "SupervisedLoss")
    assert not hasattr(medrap, "SyntheticSupervisedDatamodule")
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
                "prep.tokenizer.pretrained_model_name_or_path=stub-tokenizer",
                "prep.embedder.model_name_or_path=stub-embedder",
                f"prep.output.output_dir={tmp_path}/prepared",
            ],
        )

    assert cfg.prep.source.dataset_path == f"{tmp_path}/source"
    assert cfg.prep.document.fields == ["text"]
    assert cfg.prep.index.index_name == "retrieval"


def test_train_compose_marginalized_retrieval(tmp_path) -> None:
    with initialize_config_module(version_base=None, config_module="medrap.conf"):
        cfg = compose(
            config_name="_train",
            overrides=[
                "training/datamodule=synthetic",
                f"output_dir={tmp_path / 'marg'}",
                "marginalized_retrieval=true",
                "retrieval_encoder=key_embedding",
                "retriever.k=2",
                "training/task=marginalized_binary",
                "training/loss=marginalized_retrieval",
            ],
        )

    lightning_module = instantiate_training_module(cfg)
    assert lightning_module.loss_fn.__class__.__name__ == "MarginalizedRetrievalSupervisedLoss"
    assert lightning_module.task.__class__.__name__ == "MarginalizedBinaryClassificationTask"
    assert cfg.training.task.output_dim == 2

    batch = _example_batch()
    batch.boolean_value = torch.BoolTensor([True, False])
    out = lightning_module.model.forward(batch)
    assert out.logits.shape == (2, 2)
    targets = lightning_module.task.extract_targets(batch)
    assert lightning_module.loss_fn(out, targets).ndim == 0


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
                "output_dir=outputs/medrap-eval",
                "checkpoint_path=outputs/medrap-eval/checkpoints/last.ckpt",
            ],
        )

    model = instantiate_training_module(cfg).model

    assert isinstance(model, RetrievalAugmentedModel)
    assert isinstance(model.retriever, HFDatasetRetriever)
    out = model.forward(_example_batch())
    assert out.logits.shape == (2, 1)
    assert out.logits.dtype == torch.float32
