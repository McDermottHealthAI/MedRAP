"""CLI entrypoints for medrap."""

import os
import shutil
from pathlib import Path

import hydra
import lightning
import torch
from hydra_zen import instantiate
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from .configs import (
    instantiate_datamodule,
    instantiate_training_module,
    prepare_retrieval_dataset_from_config,
)
from .preprocess.label_creation import generate_tasks
from .preprocess.preprocessing import run_meds_pipeline


def _save_resolved(cfg: DictConfig, output_path: Path) -> None:
    resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(resolved_cfg, output_path)


def _save_resolved_config(cfg: DictConfig, output_path: Path) -> None:
    bound_cfg = _bind_trainer_paths(cfg, output_dir=output_path.parent)
    _save_resolved(bound_cfg, output_path)


def _bind_trainer_paths(cfg: DictConfig, *, output_dir: Path) -> DictConfig:
    bound_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    bound_cfg.training.trainer.default_root_dir = str(output_dir)

    logger_cfg = bound_cfg.training.trainer.get("logger")
    if isinstance(logger_cfg, DictConfig) and "save_dir" in logger_cfg:
        logger_cfg.save_dir = f"{output_dir}/loggers"
    elif OmegaConf.is_list(logger_cfg):
        for log in logger_cfg:
            if isinstance(log, DictConfig) and "save_dir" in log:
                log.save_dir = f"{output_dir}/loggers"

    callbacks_cfg = bound_cfg.training.trainer.get("callbacks")
    if OmegaConf.is_list(callbacks_cfg):
        for callback_cfg in callbacks_cfg:
            if isinstance(callback_cfg, DictConfig) and "dirpath" in callback_cfg:
                callback_cfg.dirpath = f"{output_dir}/checkpoints"

    return bound_cfg


def _find_checkpoint_path(output_dir: Path) -> Path | None:
    """Return the resume checkpoint path for a saved run directory.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     run_dir = Path(tmpdir)
        ...     _find_checkpoint_path(run_dir) is None
        True
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     run_dir = Path(tmpdir)
        ...     checkpoints = run_dir / "checkpoints"
        ...     checkpoints.mkdir()
        ...     _ = (checkpoints / "epoch=0-step=1.ckpt").write_text("older")
        ...     newer = checkpoints / "epoch=1-step=2.ckpt"
        ...     _ = newer.write_text("newer")
        ...     _find_checkpoint_path(run_dir) == newer
        True
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     run_dir = Path(tmpdir)
        ...     _ = (run_dir / "checkpoints").write_text("not a directory")
        ...     _find_checkpoint_path(run_dir)
        Traceback (most recent call last):
        NotADirectoryError: Checkpoints directory .../checkpoints is a file, not a directory.
    """
    checkpoints_dir = output_dir / "checkpoints"
    if checkpoints_dir.is_file():
        raise NotADirectoryError(f"Checkpoints directory {checkpoints_dir} is a file, not a directory.")
    if not checkpoints_dir.exists():
        return None

    last_ckpt = checkpoints_dir / "last.ckpt"
    if last_ckpt.is_file():
        return last_ckpt

    checkpoint_files = sorted(checkpoints_dir.glob("epoch=*-step=*.ckpt"))
    return checkpoint_files[-1] if checkpoint_files else None


def _validate_resume_directory(output_dir: Path, cfg: DictConfig) -> None:
    """Validate that a resumed run matches the saved config.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     run_dir = Path(tmpdir)
        ...     cfg = OmegaConf.create(
        ...         {"output_dir": str(run_dir), "training": {"trainer": {"default_root_dir": "."}}}
        ...     )
        ...     _validate_resume_directory(run_dir, cfg)
        Traceback (most recent call last):
        FileNotFoundError: No saved config found at .../config.yaml.
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     run_dir = Path(tmpdir)
        ...     run_dir.mkdir(exist_ok=True)
        ...     saved_cfg = OmegaConf.create(
        ...         {
        ...             "output_dir": str(run_dir),
        ...             "training": {"trainer": {"default_root_dir": "."}, "task": {"output_dim": 1}},
        ...         }
        ...     )
        ...     new_cfg = OmegaConf.create(
        ...         {
        ...             "output_dir": str(run_dir),
        ...             "training": {"trainer": {"default_root_dir": "."}, "task": {"output_dim": 2}},
        ...         }
        ...     )
        ...     OmegaConf.save(saved_cfg, run_dir / "config.yaml")
        ...     _validate_resume_directory(run_dir, new_cfg)
        Traceback (most recent call last):
        ValueError: Config mismatch when resuming run in ...
    """
    config_path = output_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"No saved config found at {config_path}.")

    old_cfg = _bind_trainer_paths(OmegaConf.load(config_path), output_dir=output_dir)
    new_cfg = _bind_trainer_paths(cfg, output_dir=output_dir)
    old_container = OmegaConf.to_container(old_cfg, resolve=True)
    new_container = OmegaConf.to_container(new_cfg, resolve=True)

    assert isinstance(old_container, dict)
    assert isinstance(new_container, dict)

    for key in ("do_resume", "do_overwrite", "hydra"):
        old_container.pop(key, None)
        new_container.pop(key, None)

    if old_container != new_container:
        raise ValueError(f"Config mismatch when resuming run in {output_dir}.")


def _prepare_output_dir(cfg: DictConfig) -> Path:
    """Return the configured output directory, rejecting file paths.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     output_path = Path(tmpdir) / "output"
        ...     output_path.write_text("not a directory")
        ...     cfg = OmegaConf.create({"output_dir": str(output_path)})
        ...     _prepare_output_dir(cfg)
        Traceback (most recent call last):
        NotADirectoryError: Output directory .../output is a file, not a directory.
    """
    output_dir = Path(cfg.output_dir)
    if output_dir.is_file():
        raise NotADirectoryError(f"Output directory {output_dir} is a file, not a directory.")
    return output_dir


def _setup_output_dir(output_dir: Path, cfg: DictConfig) -> Path:
    """Create or overwrite a command output directory and save configs.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     output_dir = Path(tmpdir) / "out"
        ...     cfg = OmegaConf.create({"do_overwrite": False})
        ...     returned = _setup_output_dir(output_dir, cfg)
        ...     has_config = (output_dir / "config.yaml").exists()
        ...     has_resolved = (output_dir / "resolved_config.yaml").exists()
        ...     returned == output_dir and has_config and has_resolved
        True
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     output_dir = Path(tmpdir) / "out"
        ...     output_dir.mkdir()
        ...     _ = (output_dir / "config.yaml").write_text("existing")
        ...     cfg = OmegaConf.create({"do_overwrite": False})
        ...     _setup_output_dir(output_dir, cfg)
        Traceback (most recent call last):
        FileExistsError: Output directory .../out already exists. Use `do_overwrite=true` to proceed.
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     output_dir = Path(tmpdir) / "out"
        ...     output_dir.mkdir()
        ...     stale_file = output_dir / "stale.txt"
        ...     _ = stale_file.write_text("stale")
        ...     _ = (output_dir / "config.yaml").write_text("existing")
        ...     cfg = OmegaConf.create({"do_overwrite": True})
        ...     returned = _setup_output_dir(output_dir, cfg)
        ...     returned == output_dir and not stale_file.exists()
        True
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     output_path = Path(tmpdir) / "out"
        ...     _ = output_path.write_text("not a directory")
        ...     cfg = OmegaConf.create({"do_overwrite": False})
        ...     _setup_output_dir(output_path, cfg)
        Traceback (most recent call last):
        NotADirectoryError: Output directory .../out is a file, not a directory.
    """
    if output_dir.is_file():
        raise NotADirectoryError(f"Output directory {output_dir} is a file, not a directory.")
    config_path = output_dir / "config.yaml"
    if config_path.exists():
        if cfg.do_overwrite:
            shutil.rmtree(output_dir, ignore_errors=True)
        else:
            raise FileExistsError(
                f"Output directory {output_dir} already exists. Use `do_overwrite=true` to proceed."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, config_path)
    _save_resolved(cfg, output_dir / "resolved_config.yaml")
    return output_dir


def _prepare_train_run(cfg: DictConfig) -> Path | None:
    """Create or validate a training run directory.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     output_dir = Path(tmpdir) / "train"
        ...     output_dir.mkdir()
        ...     stale_file = output_dir / "stale.txt"
        ...     _ = stale_file.write_text("stale")
        ...     _ = (output_dir / "config.yaml").write_text("existing")
        ...     cfg = OmegaConf.create(
        ...         {
        ...             "output_dir": str(output_dir),
        ...             "do_overwrite": True,
        ...             "do_resume": False,
        ...             "training": {"trainer": {"default_root_dir": "."}},
        ...         }
        ...     )
        ...     _prepare_train_run(cfg) is None
        True
        >>> stale_file.exists()
        False
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     output_dir = Path(tmpdir) / "train"
        ...     output_dir.mkdir()
        ...     cfg = OmegaConf.create(
        ...         {
        ...             "output_dir": str(output_dir),
        ...             "do_overwrite": False,
        ...             "do_resume": True,
        ...             "training": {"trainer": {"default_root_dir": "."}},
        ...         }
        ...     )
        ...     OmegaConf.save(cfg, output_dir / "config.yaml")
        ...     _prepare_train_run(cfg)
        Traceback (most recent call last):
        FileNotFoundError: No checkpoint found to resume from in ...
    """
    output_dir = _prepare_output_dir(cfg)
    config_path = output_dir / "config.yaml"
    ckpt_path = None

    if config_path.exists():
        if cfg.do_overwrite:
            shutil.rmtree(output_dir, ignore_errors=True)
        elif cfg.do_resume:
            _validate_resume_directory(output_dir, cfg)
            ckpt_path = _find_checkpoint_path(output_dir)
            if ckpt_path is None:
                raise FileNotFoundError(f"No checkpoint found to resume from in {output_dir}.")
        else:
            raise FileExistsError(
                f"Output directory {output_dir} already contains a saved MedRAP run. "
                "Use `do_overwrite=true` or `do_resume=true` to proceed."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        OmegaConf.save(cfg, config_path)
        _save_resolved_config(cfg, output_dir / "resolved_config.yaml")

    return ckpt_path


def _prepare_eval_run(cfg: DictConfig) -> Path:
    output_dir = _prepare_output_dir(cfg)
    config_path = output_dir / "config.yaml"

    if config_path.exists():
        raise FileExistsError(
            f"Output directory {output_dir} already contains a saved MedRAP eval run. "
            "Use a different `output_dir`."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, config_path)
    _save_resolved_config(cfg, output_dir / "resolved_config.yaml")
    return output_dir


def _prepare_retrieval_dataset_run(cfg: DictConfig) -> Path:
    return _setup_output_dir(Path(cfg.prep.output.output_dir), cfg)


def _ensure_lightning_csv_log_dirs(trainer: object) -> None:
    """Create CSVLogger ``log_dir`` trees before the first metrics flush.

    Lightning's logger connector calls ``save()`` after every ``log_metrics`` on
    each logger. PyTorch Lightning's CSV backend can attempt to open
    ``metrics.csv`` when the versioned directory is not yet visible on disk
    (for example on some shared filesystems), which raises ``FileNotFoundError``.
    Touching ``log_dir`` here pins the auto-assigned version and ensures the
    directory exists using the process-local filesystem API.

    Examples:
        >>> import tempfile
        >>> from lightning.pytorch import Trainer
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     lg = CSVLogger(save_dir=tmp, name="csv")
        ...     tr = Trainer(logger=lg, max_epochs=0, accelerator="cpu", devices=1)
        ...     _ensure_lightning_csv_log_dirs(tr)
        ...     os.path.isdir(lg.log_dir)
        True
    """
    for lg in getattr(trainer, "loggers", []):
        if isinstance(lg, CSVLogger):
            os.makedirs(lg.log_dir, exist_ok=True)


def _copy_best_checkpoint(trainer: object, output_dir: Path) -> None:
    """Copy the best checkpoint into the run root when available.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     _copy_best_checkpoint(SimpleNamespace(checkpoint_callback=None), Path(tmpdir))
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     _copy_best_checkpoint(
        ...         SimpleNamespace(checkpoint_callback=SimpleNamespace(best_model_path="")),
        ...         Path(tmpdir),
        ...     )
    """
    checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
    if checkpoint_callback is None:
        return

    best_model_path = getattr(checkpoint_callback, "best_model_path", "")
    if not best_model_path:
        return

    best_ckpt = Path(best_model_path)
    if best_ckpt.is_file():
        shutil.copyfile(best_ckpt, output_dir / "best_model.ckpt")


def _load_training_module_checkpoint(cfg: DictConfig, checkpoint_path: str) -> object:
    module = instantiate_training_module(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    module.load_state_dict(checkpoint["state_dict"])
    return module


def _run_train(cfg: DictConfig) -> int:
    print(OmegaConf.to_yaml(cfg))
    lightning.seed_everything(cfg.seed, workers=True)
    output_dir = _prepare_output_dir(cfg)
    ckpt_path = _prepare_train_run(cfg)
    module = instantiate_training_module(cfg)
    bound_cfg = _bind_trainer_paths(cfg, output_dir=output_dir)
    trainer = instantiate(bound_cfg.training.trainer)
    _ensure_lightning_csv_log_dirs(trainer)
    datamodule = instantiate_datamodule(cfg)

    trainer.fit(module, datamodule=datamodule, ckpt_path=str(ckpt_path) if ckpt_path else None)
    _copy_best_checkpoint(trainer, output_dir)
    return 0


def _run_eval(cfg: DictConfig) -> int:
    print(OmegaConf.to_yaml(cfg))
    output_dir = _prepare_eval_run(cfg)
    checkpoint_path = cfg.checkpoint_path
    if not checkpoint_path:
        raise ValueError("checkpoint_path must be set for medrap-eval.")

    module = _load_training_module_checkpoint(cfg, checkpoint_path)
    bound_cfg = _bind_trainer_paths(cfg, output_dir=output_dir)
    trainer = instantiate(bound_cfg.training.trainer)
    _ensure_lightning_csv_log_dirs(trainer)
    datamodule = instantiate_datamodule(cfg)

    if cfg.eval_mode == "validate":
        trainer.validate(module, datamodule=datamodule)
        return 0
    if cfg.eval_mode == "test":
        trainer.test(module, datamodule=datamodule)
        return 0
    raise ValueError(f"eval_mode must be 'validate' or 'test'; got {cfg.eval_mode!r}")


def _run_prepare_retrieval_dataset(cfg: DictConfig) -> int:
    print(OmegaConf.to_yaml(cfg))
    _prepare_retrieval_dataset_run(cfg)
    output_path = prepare_retrieval_dataset_from_config(cfg)
    print(f"Prepared retrieval dataset saved to {output_path}")
    return 0


@hydra.main(version_base=None, config_path="conf", config_name="_preprocess")
def preprocess_main(cfg: DictConfig) -> None:
    """Run the Hydra-native MEDS preprocessing and task-generation entrypoint."""
    print(OmegaConf.to_yaml(cfg))
    _setup_output_dir(Path(cfg.output_dir), cfg)

    if cfg.tensorized_dir is None:
        intermediate_dir, tensorized_dir = run_meds_pipeline(
            cfg.meds_data_dir,
            cfg.output_dir,
            min_subjects_per_code=cfg.min_subjects_per_code,
            min_events_per_subject=cfg.min_events_per_subject,
        )
        print(f"Intermediate MEDS dataset saved to {intermediate_dir}")
        print(f"Tensorized dataset saved to {tensorized_dir}")
        task_source_dir = intermediate_dir
    else:
        print(f"Skipping preprocessing; using existing tensorized dir: {cfg.tensorized_dir}")
        task_source_dir = Path(cfg.meds_data_dir)

    tasks_out = generate_tasks(
        meds_data_dir=task_source_dir,
        output_dir=Path(cfg.output_dir) / "tasks",
        num_tasks=cfg.num_tasks,
        horizon_days=cfg.horizon_days,
        min_history_days=cfg.min_history_days,
        seed=cfg.seed,
    )
    print(f"Task labels saved to {tasks_out}")


@hydra.main(version_base=None, config_path="conf", config_name="_train")
def train_main(cfg: DictConfig) -> int:
    """Run the Hydra-native training entrypoint."""
    return _run_train(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="_eval")
def eval_main(cfg: DictConfig) -> int:
    """Run the Hydra-native evaluation entrypoint."""
    return _run_eval(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="_prepare_retrieval_dataset")
def prepare_retrieval_dataset_main(cfg: DictConfig) -> int:
    """Run the Hydra-native retrieval-dataset preparation entrypoint."""
    return _run_prepare_retrieval_dataset(cfg)
