import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

import medrap.cli as cli
from medrap.cli import eval_main, prepare_retrieval_dataset_main, preprocess_main, train_main

TINY_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


def _assert_cli_failure(
    call, *, allowed_exceptions: tuple[type[BaseException], ...], expected_message: str | None = None
) -> None:
    with pytest.raises(allowed_exceptions) as exc_info:
        call()

    if expected_message is not None and not isinstance(exc_info.value, SystemExit):
        assert expected_message in str(exc_info.value)


def _run_hydra_entrypoint(entrypoint, prog: str, overrides: list[str]) -> int:
    old_argv = sys.argv
    try:
        sys.argv = [prog, *overrides]
        result = entrypoint()
        return int(result) if isinstance(result, int) else 0
    finally:
        sys.argv = old_argv


def _run_train_cli(overrides: list[str]) -> int:
    return _run_hydra_entrypoint(train_main, "medrap-train", overrides)


def _run_eval_cli(overrides: list[str]) -> int:
    return _run_hydra_entrypoint(eval_main, "medrap-eval", overrides)


def _run_prepare_retrieval_dataset_cli(overrides: list[str]) -> int:
    return _run_hydra_entrypoint(
        prepare_retrieval_dataset_main, "medrap-prepare-retrieval-dataset", overrides
    )


def _run_preprocess_cli(overrides: list[str]) -> int:
    return _run_hydra_entrypoint(preprocess_main, "medrap-preprocess", overrides)


def test_medrap_train_entrypoint_runs_with_overrides(tmp_path) -> None:
    output_dir = tmp_path / "train"

    assert _run_train_cli([f"output_dir={output_dir}", "training/datamodule=synthetic"]) == 0
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "resolved_config.yaml").exists()
    assert (output_dir / "checkpoints" / "last.ckpt").exists()
    assert (output_dir / "best_model.ckpt").exists()


def test_medrap_eval_entrypoint_runs_with_overrides(tmp_path) -> None:
    output_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    assert _run_train_cli([f"output_dir={output_dir}", "training/datamodule=synthetic"]) == 0

    assert (
        _run_eval_cli(
            [
                f"output_dir={eval_dir}",
                f"checkpoint_path={checkpoint_path}",
                "training/datamodule=synthetic",
            ]
        )
        == 0
    )
    assert (eval_dir / "config.yaml").exists()
    assert (eval_dir / "resolved_config.yaml").exists()


def test_flat_entrypoint_scripts_are_registered() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    medrap_scripts = pyproject["project"]["scripts"]

    assert "medrap" not in medrap_scripts
    assert medrap_scripts["medrap-train"] == "medrap.cli:train_main"
    assert medrap_scripts["medrap-eval"] == "medrap.cli:eval_main"
    assert medrap_scripts["medrap-prepare-retrieval-dataset"] == "medrap.cli:prepare_retrieval_dataset_main"
    assert medrap_scripts["medrap-preprocess"] == "medrap.cli:preprocess_main"


def test_prepare_retrieval_dataset_entrypoint_runs_with_hydra_overrides(monkeypatch, tmp_path) -> None:
    captured = {}

    def _fake_prepare_retrieval_dataset_from_config(cfg):
        captured["source"] = cfg.prep.source.dataset_path
        captured["fields"] = cfg.prep.document.fields
        captured["tokenizer"] = cfg.prep.tokenizer.pretrained_model_name_or_path
        captured["embedder"] = cfg.prep.embedder.model_name_or_path
        captured["device"] = cfg.prep.embedder.device
        captured["output_dir"] = cfg.prep.output.output_dir
        captured["max_length"] = cfg.prep.index.max_length
        return output_dir

    monkeypatch.setattr(
        "medrap.cli.prepare_retrieval_dataset_from_config", _fake_prepare_retrieval_dataset_from_config
    )

    source_dir = tmp_path / "source"
    output_dir = tmp_path / "prepared"

    assert (
        _run_prepare_retrieval_dataset_cli(
            [
                "prep/source=load_from_disk",
                f"prep.source.dataset_path={source_dir}",
                "prep.document.fields=[text]",
                f"prep.tokenizer.pretrained_model_name_or_path={TINY_SENTENCE_TRANSFORMER_MODEL}",
                f"prep.embedder.model_name_or_path={TINY_SENTENCE_TRANSFORMER_MODEL}",
                "prep.embedder.device=cpu",
                f"prep.output.output_dir={output_dir}",
                "prep.index.max_length=3",
            ]
        )
        == 0
    )
    assert captured == {
        "source": str(source_dir),
        "fields": ["text"],
        "tokenizer": TINY_SENTENCE_TRANSFORMER_MODEL,
        "embedder": TINY_SENTENCE_TRANSFORMER_MODEL,
        "device": "cpu",
        "output_dir": str(output_dir),
        "max_length": 3,
    }
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "resolved_config.yaml").exists()


def test_eval_entrypoint_requires_checkpoint_path(tmp_path) -> None:
    _assert_cli_failure(
        lambda: _run_eval_cli([f"output_dir={tmp_path / 'eval'}", "training/datamodule=synthetic"]),
        allowed_exceptions=(SystemExit, MissingMandatoryValue, ValueError),
        expected_message="checkpoint_path",
    )


def test_prepare_train_run_overwrite_removes_stale_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "train"
    output_dir.mkdir()
    stale = output_dir / "stale.txt"
    stale.write_text("old")
    cfg = OmegaConf.create(
        {
            "output_dir": str(output_dir),
            "do_overwrite": True,
            "do_resume": False,
            "training": {"trainer": {"default_root_dir": "."}},
        }
    )
    OmegaConf.save(cfg, output_dir / "config.yaml")
    assert cli._prepare_train_run(cfg) is None
    assert not stale.exists()


def test_train_entrypoint_refuses_existing_output_dir_without_flags(tmp_path) -> None:
    output_dir = tmp_path / "train"
    assert _run_train_cli([f"output_dir={output_dir}", "training/datamodule=synthetic"]) == 0

    _assert_cli_failure(
        lambda: _run_train_cli([f"output_dir={output_dir}", "training/datamodule=synthetic"]),
        allowed_exceptions=(SystemExit, FileExistsError),
        expected_message="already contains a saved MedRAP run",
    )


def test_train_entrypoint_supports_resume_from_output_dir(tmp_path) -> None:
    output_dir = tmp_path / "train"
    assert _run_train_cli([f"output_dir={output_dir}", "training/datamodule=synthetic"]) == 0

    assert (
        _run_train_cli([f"output_dir={output_dir}", "training/datamodule=synthetic", "do_resume=true"]) == 0
    )


def test_eval_entrypoint_supports_test_mode(tmp_path) -> None:
    output_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    assert _run_train_cli([f"output_dir={output_dir}", "training/datamodule=synthetic"]) == 0

    assert (
        _run_eval_cli(
            [
                f"output_dir={eval_dir}",
                f"checkpoint_path={checkpoint_path}",
                "eval_mode=test",
                "training/datamodule=synthetic",
            ]
        )
        == 0
    )


def test_eval_entrypoint_refuses_existing_output_dir(tmp_path) -> None:
    output_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    assert _run_train_cli([f"output_dir={output_dir}", "training/datamodule=synthetic"]) == 0
    assert (
        _run_eval_cli(
            [
                f"output_dir={eval_dir}",
                f"checkpoint_path={checkpoint_path}",
                "training/datamodule=synthetic",
            ]
        )
        == 0
    )

    _assert_cli_failure(
        lambda: _run_eval_cli(
            [
                f"output_dir={eval_dir}",
                f"checkpoint_path={checkpoint_path}",
                "training/datamodule=synthetic",
            ]
        ),
        allowed_exceptions=(SystemExit, FileExistsError),
        expected_message="already contains a saved MedRAP eval run",
    )


def test_ensure_lightning_csv_log_dirs_creates_version_directory(tmp_path) -> None:
    log_root = tmp_path / "loggers"
    log_root.mkdir()
    csv_logger = CSVLogger(save_dir=str(log_root), name="csv")
    trainer = Trainer(logger=csv_logger, max_epochs=0, accelerator="cpu", devices=1)
    cli._ensure_lightning_csv_log_dirs(trainer)
    assert csv_logger.log_dir
    assert (tmp_path / "loggers" / "csv" / "version_0").is_dir()


def test_run_eval_requires_checkpoint_path_before_loading(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path / "eval"),
            "checkpoint_path": "",
            "eval_mode": "validate",
            "training": {"trainer": {"default_root_dir": "."}},
        }
    )

    with pytest.raises(ValueError, match="checkpoint_path must be set"):
        cli._run_eval(cfg)


def test_bind_trainer_paths_sets_single_logger_save_dir(tmp_path) -> None:
    output_dir = tmp_path / "run"
    cfg = OmegaConf.create(
        {
            "training": {
                "trainer": {
                    "default_root_dir": ".",
                    "logger": {
                        "_target_": "lightning.pytorch.loggers.csv_logs.CSVLogger",
                        "save_dir": "/placeholder/loggers",
                        "name": "csv",
                    },
                }
            }
        }
    )

    bound = cli._bind_trainer_paths(cfg, output_dir=output_dir)

    assert bound.training.trainer.default_root_dir == str(output_dir)
    assert bound.training.trainer.logger.save_dir == str(output_dir / "loggers")


def test_bind_trainer_paths_skips_logger_dict_without_save_dir(tmp_path) -> None:
    output_dir = tmp_path / "run"
    cfg = OmegaConf.create(
        {
            "training": {
                "trainer": {
                    "default_root_dir": ".",
                    "logger": {
                        "_target_": "lightning.pytorch.loggers.TensorBoardLogger",
                        "name": "tb",
                    },
                }
            }
        }
    )

    bound = cli._bind_trainer_paths(cfg, output_dir=output_dir)

    assert "save_dir" not in bound.training.trainer.logger


def test_bind_trainer_paths_sets_save_dir_for_each_list_logger(tmp_path) -> None:
    output_dir = tmp_path / "run"
    cfg = OmegaConf.create(
        {
            "training": {
                "trainer": {
                    "default_root_dir": ".",
                    "logger": [
                        {
                            "_target_": "lightning.pytorch.loggers.csv_logs.CSVLogger",
                            "save_dir": "/old/csv",
                            "name": "csv",
                        },
                        {
                            "_target_": "lightning.pytorch.loggers.WandbLogger",
                            "project": "p",
                            "save_dir": "/old/wandb",
                        },
                    ],
                }
            }
        }
    )

    bound = cli._bind_trainer_paths(cfg, output_dir=output_dir)

    assert bound.training.trainer.logger[0].save_dir == str(output_dir / "loggers")
    assert bound.training.trainer.logger[1].save_dir == str(output_dir / "loggers")


def test_bind_trainer_paths_list_logger_without_save_dir_left_unchanged(tmp_path) -> None:
    output_dir = tmp_path / "run"
    cfg = OmegaConf.create(
        {
            "training": {
                "trainer": {
                    "default_root_dir": ".",
                    "logger": [
                        {
                            "_target_": "lightning.pytorch.loggers.csv_logs.CSVLogger",
                            "save_dir": "/old/csv",
                            "name": "csv",
                        },
                        {"_target_": "SomeLogger", "name": "no_save_dir"},
                    ],
                }
            }
        }
    )

    bound = cli._bind_trainer_paths(cfg, output_dir=output_dir)

    assert bound.training.trainer.logger[0].save_dir == str(output_dir / "loggers")
    assert "save_dir" not in bound.training.trainer.logger[1]


def test_find_checkpoint_path_variants(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir()
    assert cli._find_checkpoint_path(run_dir) is None

    last = ckpt_dir / "last.ckpt"
    last.write_text("x")
    assert cli._find_checkpoint_path(run_dir) == last

    last.unlink()
    old = ckpt_dir / "epoch=0-step=1.ckpt"
    new = ckpt_dir / "epoch=1-step=2.ckpt"
    old.write_text("a")
    new.write_text("b")
    assert cli._find_checkpoint_path(run_dir) == new

    for p in ckpt_dir.iterdir():
        p.unlink()
    ckpt_dir.rmdir()
    checkpoints_file = run_dir / "checkpoints"
    checkpoints_file.write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        cli._find_checkpoint_path(run_dir)


def test_prepare_output_dir_rejects_file_path(tmp_path: Path) -> None:
    f = tmp_path / "out"
    f.write_text("x")
    cfg = OmegaConf.create({"output_dir": str(f)})
    with pytest.raises(NotADirectoryError):
        cli._prepare_output_dir(cfg)


def test_prepare_train_run_resume_without_checkpoint_raises(tmp_path: Path) -> None:
    output_dir = tmp_path / "train"
    output_dir.mkdir()
    cfg = OmegaConf.create(
        {
            "output_dir": str(output_dir),
            "do_overwrite": False,
            "do_resume": True,
            "training": {"trainer": {"default_root_dir": "."}},
        }
    )
    OmegaConf.save(cfg, output_dir / "config.yaml")
    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        cli._prepare_train_run(cfg)


def test_validate_resume_directory_missing_config_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = OmegaConf.create({"output_dir": str(run_dir), "training": {"trainer": {"default_root_dir": "."}}})
    with pytest.raises(FileNotFoundError, match="No saved config"):
        cli._validate_resume_directory(run_dir, cfg)


def test_validate_resume_directory_mismatch_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    saved = OmegaConf.create(
        {
            "output_dir": str(run_dir),
            "training": {"trainer": {"default_root_dir": "."}, "task": {"output_dim": 1}},
        }
    )
    new_cfg = OmegaConf.create(
        {
            "output_dir": str(run_dir),
            "training": {"trainer": {"default_root_dir": "."}, "task": {"output_dim": 2}},
        }
    )
    OmegaConf.save(saved, run_dir / "config.yaml")
    with pytest.raises(ValueError, match="Config mismatch"):
        cli._validate_resume_directory(run_dir, new_cfg)


def test_bind_trainer_paths_sets_callback_dirpath(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    cfg = OmegaConf.create(
        {
            "training": {
                "trainer": {
                    "default_root_dir": ".",
                    "callbacks": [
                        {
                            "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                            "dirpath": "/old/ckpt",
                        }
                    ],
                }
            }
        }
    )
    bound = cli._bind_trainer_paths(cfg, output_dir=output_dir)
    assert bound.training.trainer.callbacks[0].dirpath == str(output_dir / "checkpoints")


def test_copy_best_checkpoint_copies_when_present(tmp_path: Path) -> None:
    src = tmp_path / "best.ckpt"
    src.write_text("weights")
    cb = SimpleNamespace(best_model_path=str(src))
    trainer = SimpleNamespace(checkpoint_callback=cb)
    out = tmp_path / "run"
    out.mkdir()
    cli._copy_best_checkpoint(trainer, out)
    assert (out / "best_model.ckpt").read_text() == "weights"


def test_copy_best_checkpoint_noops_without_callback_or_path(tmp_path: Path) -> None:
    cli._copy_best_checkpoint(SimpleNamespace(checkpoint_callback=None), tmp_path)
    empty_best = SimpleNamespace(best_model_path="")
    cli._copy_best_checkpoint(SimpleNamespace(checkpoint_callback=empty_best), tmp_path)


def test_run_eval_rejects_unknown_eval_mode(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    class StubTrainer:
        def validate(self, module, datamodule=None) -> None:  # pragma: no cover
            raise AssertionError("validate should not run")

        def test(self, module, datamodule=None) -> None:  # pragma: no cover
            raise AssertionError("test should not run")

    monkeypatch.setattr(cli, "_load_training_module_checkpoint", lambda cfg, checkpoint_path: object())
    monkeypatch.setattr(cli, "instantiate", lambda trainer_cfg: StubTrainer())
    monkeypatch.setattr(cli, "instantiate_datamodule", lambda cfg: object())
    cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path / "eval"),
            "checkpoint_path": str(tmp_path / "model.ckpt"),
            "eval_mode": "predict",
            "training": {"trainer": {"default_root_dir": "."}},
        }
    )

    with pytest.raises(ValueError, match="eval_mode must be 'validate' or 'test'"):
        cli._run_eval(cfg)


def test_preprocess_entrypoint_runs_with_hydra_overrides(monkeypatch, tmp_path) -> None:
    captured = {}

    def _fake_generate_tasks(meds_data_dir, output_dir, *, num_tasks, horizon_days, min_history_days, seed):
        captured["meds_data_dir"] = str(meds_data_dir)
        captured["num_tasks"] = num_tasks
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return Path(output_dir)

    monkeypatch.setattr("medrap.cli.generate_tasks", _fake_generate_tasks)

    meds_data_dir = tmp_path / "raw"
    output_dir = tmp_path / "out"
    tensorized_dir = tmp_path / "tensorized"

    assert (
        _run_preprocess_cli(
            [
                f"meds_data_dir={meds_data_dir}",
                f"output_dir={output_dir}",
                f"tensorized_dir={tensorized_dir}",
                "num_tasks=5",
            ]
        )
        == 0
    )
    assert captured["meds_data_dir"] == str(meds_data_dir)
    assert captured["num_tasks"] == 5
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "resolved_config.yaml").exists()


def test_preprocess_entrypoint_runs_without_tensorized_dir(monkeypatch, tmp_path) -> None:
    captured = {}

    def _fake_run_meds_pipeline(meds_data_dir, output_dir, *, min_subjects_per_code, min_events_per_subject):
        inter = Path(output_dir) / "intermediate"
        tens = Path(output_dir) / "tensorized"
        inter.mkdir(parents=True, exist_ok=True)
        tens.mkdir(parents=True, exist_ok=True)
        captured["meds_data_dir"] = str(meds_data_dir)
        return inter, tens

    def _fake_generate_tasks(meds_data_dir, output_dir, *, num_tasks, horizon_days, min_history_days, seed):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return Path(output_dir)

    monkeypatch.setattr("medrap.cli.run_meds_pipeline", _fake_run_meds_pipeline)
    monkeypatch.setattr("medrap.cli.generate_tasks", _fake_generate_tasks)

    output_dir = tmp_path / "out"
    assert (
        _run_preprocess_cli(
            [
                f"meds_data_dir={tmp_path / 'raw'}",
                f"output_dir={output_dir}",
            ]
        )
        == 0
    )
    assert captured["meds_data_dir"] == str(tmp_path / "raw")
    assert (output_dir / "config.yaml").exists()


def test_run_meds_pipeline_creates_dirs_and_invokes_subprocesses(monkeypatch, tmp_path) -> None:
    from medrap.preprocess.preprocessing import run_meds_pipeline

    calls = []

    def _fake_subprocess_run(cmd, *, env=None, check=False):
        calls.append(cmd[0])

    monkeypatch.setattr("medrap.preprocess.preprocessing.subprocess.run", _fake_subprocess_run)

    inter, tens = run_meds_pipeline(
        tmp_path / "raw",
        tmp_path / "out",
        min_subjects_per_code=10,
        min_events_per_subject=5,
    )
    assert inter == tmp_path / "out" / "intermediate"
    assert tens == tmp_path / "out" / "tensorized"
    assert inter.exists() and tens.exists()
    assert calls == ["MEDS_transform-pipeline", "MTD_preprocess"]
