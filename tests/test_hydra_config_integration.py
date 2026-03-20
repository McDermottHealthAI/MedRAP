from hydra import compose, initialize_config_module
from torch import nn

from medrap.configs import instantiate_datamodule, instantiate_training_module
from medrap.lightning_module import MedRAPSupervisedLightningModule
from medrap.model import RetrievalAugmentedModel
from medrap.task import SupervisedTask


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
    datamodule = instantiate_datamodule(cfg)

    assert isinstance(lightning_module, MedRAPSupervisedLightningModule)
    assert isinstance(lightning_module.model, RetrievalAugmentedModel)
    assert isinstance(lightning_module.task, nn.Module)
    assert lightning_module.loss_fn.__class__.__name__ == "BinaryClassificationLoss"
    assert datamodule.__class__.__name__ == "SyntheticSupervisedDatamodule"
    assert cfg.training.task.output_dim == 1
    assert cfg.head.out_dim == cfg.training.task.output_dim


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
    datamodule = instantiate_datamodule(cfg)

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

    datamodule = instantiate_datamodule(cfg)

    assert datamodule.__class__.__name__ == "Datamodule"


def test_supervised_task_is_not_exported_from_package_root() -> None:
    import medrap

    assert not hasattr(medrap, "SupervisedTask")
    assert not hasattr(medrap, "SupervisedLoss")
    assert not hasattr(medrap, "SyntheticSupervisedDatamodule")
    assert SupervisedTask.__name__ == "SupervisedTask"
