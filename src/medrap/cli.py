"""CLI entrypoints for medrap."""

import argparse
import sys
from collections.abc import Sequence

import hydra
from omegaconf import DictConfig, OmegaConf

from .configs import prepare_retrieval_dataset_from_config


def _run_cfg(cfg: DictConfig) -> int:
    print(OmegaConf.to_yaml(cfg))
    return 0


@hydra.main(version_base=None, config_path="conf", config_name="_train")
def _train_hydra(cfg: DictConfig) -> int:
    return _run_cfg(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="_eval")
def _eval_hydra(cfg: DictConfig) -> int:
    return _run_cfg(cfg)


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
