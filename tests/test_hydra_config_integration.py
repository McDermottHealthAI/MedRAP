import torch
from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from meds_torchdata import MEDSTorchBatch
from torch import nn

from medrap.configs import RAPAppConfig, instantiate_training_module
from medrap.lightning_module import MedRAPSupervisedLightningModule
from medrap.model import RetrievalAugmentedModel
from medrap.retrievers import InMemoryRetriever
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
    assert cfg.training.task.output_dim == 1
    assert cfg.head.out_dim == cfg.training.task.output_dim


def test_eval_config_composes_training_layer() -> None:
    with initialize_config_module(version_base=None, config_module="medrap.conf"):
        cfg = compose(config_name="_eval")

    lightning_module = instantiate_training_module(cfg)

    assert isinstance(lightning_module, MedRAPSupervisedLightningModule)
    assert isinstance(lightning_module.model, RetrievalAugmentedModel)
    assert isinstance(lightning_module.task, nn.Module)
    assert cfg.head.out_dim == cfg.training.task.output_dim


def test_app_config_registers_with_hydra_config_store() -> None:
    RAPAppConfig.add_to_config_store(group="medrap")
    cs = ConfigStore.instance()

    assert "medrap" in cs.repo
    assert "RAPAppConfig.yaml" in cs.repo["medrap"]


def test_supervised_task_is_not_exported_from_package_root() -> None:
    import medrap

    assert not hasattr(medrap, "SupervisedTask")
    assert SupervisedTask.__name__ == "SupervisedTask"
