"""CLI entrypoints for medrap."""

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import hydra
import torch
from hydra_zen import instantiate
from omegaconf import DictConfig, OmegaConf

from .configs import (
    instantiate_datamodule,
    instantiate_training_module,
    prepare_retrieval_dataset_from_config,
)


def _run_cfg(cfg: DictConfig) -> int:
    print(OmegaConf.to_yaml(cfg))
    return 0


def _save_resolved_config(cfg: DictConfig, output_path: Path) -> None:
    bound_cfg = _bind_trainer_paths(cfg, output_dir=output_path.parent)
    resolved_cfg = OmegaConf.create(OmegaConf.to_container(bound_cfg, resolve=True))
    OmegaConf.save(resolved_cfg, output_path)


def _bind_trainer_paths(cfg: DictConfig, *, output_dir: Path) -> DictConfig:
    bound_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    bound_cfg.training.trainer.default_root_dir = str(output_dir)

    logger_cfg = bound_cfg.training.trainer.get("logger")
    if isinstance(logger_cfg, DictConfig):
        logger_cfg.save_dir = f"{output_dir}/loggers"

    callbacks_cfg = bound_cfg.training.trainer.get("callbacks")
    if OmegaConf.is_list(callbacks_cfg):
        for callback_cfg in callbacks_cfg:
            if isinstance(callback_cfg, DictConfig) and "dirpath" in callback_cfg:
                callback_cfg.dirpath = f"{output_dir}/checkpoints"

    return bound_cfg


def _instantiate_trainer(cfg: DictConfig, *, output_dir: Path) -> object:
    bound_cfg = _bind_trainer_paths(cfg, output_dir=output_dir)
    return instantiate(bound_cfg.training.trainer)


def _find_checkpoint_path(output_dir: Path) -> Path | None:
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
    output_dir = Path(cfg.output_dir)
    if output_dir.is_file():
        raise NotADirectoryError(f"Output directory {output_dir} is a file, not a directory.")
    return output_dir


def _prepare_train_run(cfg: DictConfig) -> Path | None:
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


def _copy_best_checkpoint(trainer: object, output_dir: Path) -> None:
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
    output_dir = _prepare_output_dir(cfg)
    ckpt_path = _prepare_train_run(cfg)
    module = instantiate_training_module(cfg)
    trainer = _instantiate_trainer(cfg, output_dir=output_dir)
    datamodule = instantiate_datamodule(cfg)

    trainer.fit(module, datamodule=datamodule, ckpt_path=str(ckpt_path) if ckpt_path else None)
    _copy_best_checkpoint(trainer, output_dir)
    return 0


def _run_eval(cfg: DictConfig) -> int:
    print(OmegaConf.to_yaml(cfg))
    output_dir = _prepare_eval_run(cfg)
    checkpoint_path = cfg.checkpoint_path
    if not checkpoint_path:
        raise ValueError("checkpoint_path must be set for medrap eval.")

    module = _load_training_module_checkpoint(cfg, checkpoint_path)
    trainer = _instantiate_trainer(cfg, output_dir=output_dir)
    datamodule = instantiate_datamodule(cfg)

    if cfg.eval_mode == "validate":
        trainer.validate(module, datamodule=datamodule)
        return 0
    if cfg.eval_mode == "test":
        trainer.test(module, datamodule=datamodule)
        return 0
    raise ValueError(f"eval_mode must be 'validate' or 'test'; got {cfg.eval_mode!r}")


@hydra.main(version_base=None, config_path="conf", config_name="_train")
def _train_hydra(cfg: DictConfig) -> int:
    return _run_train(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="_eval")
def _eval_hydra(cfg: DictConfig) -> int:
    return _run_eval(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="_prepare_retrieval_dataset")
def _prepare_retrieval_dataset_hydra(cfg: DictConfig) -> int:
    prepare_retrieval_dataset_from_config(cfg)
    return 0


def train_main(overrides: Sequence[str] | None = None) -> int:
    """Run the Hydra-native train entrypoint."""
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0] if old_argv else "medrap-train", *(list(overrides or []))]
        result = _train_hydra()
        return int(result) if isinstance(result, int) else 0
    finally:
        sys.argv = old_argv


def eval_main(overrides: Sequence[str] | None = None) -> int:
    """Run the Hydra-native eval entrypoint."""
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0] if old_argv else "medrap-eval", *(list(overrides or []))]
        result = _eval_hydra()
        return int(result) if isinstance(result, int) else 0
    finally:
        sys.argv = old_argv


def prepare_retrieval_dataset_main(overrides: Sequence[str] | None = None) -> int:
    """Run the Hydra-native retrieval dataset preparation entrypoint."""
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0] if old_argv else "medrap-prepare-retrieval-dataset", *(list(overrides or []))]
        result = _prepare_retrieval_dataset_hydra()
        return int(result) if isinstance(result, int) else 0
    finally:
        sys.argv = old_argv


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch medrap subcommands to Hydra-native entrypoints."""
    parser = argparse.ArgumentParser(prog="medrap")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for cmd in ("train", "eval", "prepare-retrieval-dataset"):
        sub = subparsers.add_parser(cmd)
        sub.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. retriever.k=2")

    args = parser.parse_args(argv)
    if args.command == "train":
        return train_main(args.overrides)
    if args.command == "prepare-retrieval-dataset":
        return prepare_retrieval_dataset_main(args.overrides)
    return eval_main(args.overrides)
