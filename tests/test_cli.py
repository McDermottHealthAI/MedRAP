import pytest
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

import medrap.cli as cli
from medrap.cli import eval_main, main, prepare_retrieval_dataset_main, train_main

TINY_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


def _assert_cli_failure(
    call, *, allowed_exceptions: tuple[type[BaseException], ...], expected_message: str | None = None
) -> None:
    with pytest.raises(allowed_exceptions) as exc_info:
        call()

    if expected_message is not None and not isinstance(exc_info.value, SystemExit):
        assert expected_message in str(exc_info.value)


def test_medrap_train_cli_runs_with_overrides(tmp_path) -> None:
    output_dir = tmp_path / "train"

    assert main(["train", f"output_dir={output_dir}", "training/datamodule=synthetic"]) == 0
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "resolved_config.yaml").exists()
    assert (output_dir / "checkpoints" / "last.ckpt").exists()
    assert (output_dir / "best_model.ckpt").exists()


def test_medrap_eval_cli_runs_with_overrides(tmp_path) -> None:
    output_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    assert train_main([f"output_dir={output_dir}", "training/datamodule=synthetic"]) == 0

    assert (
        main(
            [
                "eval",
                f"output_dir={eval_dir}",
                f"checkpoint_path={checkpoint_path}",
                "training/datamodule=synthetic",
            ]
        )
        == 0
    )
    assert (eval_dir / "config.yaml").exists()
    assert (eval_dir / "resolved_config.yaml").exists()


def test_train_entrypoint_runs_with_hydra_overrides(tmp_path) -> None:
    output_dir = tmp_path / "train"

    assert train_main([f"output_dir={output_dir}", "training/datamodule=synthetic"]) == 0
    assert (output_dir / "checkpoints" / "last.ckpt").exists()


def test_medrap_prepare_retrieval_dataset_dispatches(monkeypatch) -> None:
    called: list[list[str]] = []

    def _fake_main(overrides: list[str]) -> int:
        called.append(overrides)
        return 0

    monkeypatch.setattr("medrap.cli.prepare_retrieval_dataset_main", _fake_main)

    assert main(["prepare-retrieval-dataset", "prep.output.output_dir=tmp"]) == 0
    assert called == [["prep.output.output_dir=tmp"]]


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

    monkeypatch.setattr(
        "medrap.cli.prepare_retrieval_dataset_from_config", _fake_prepare_retrieval_dataset_from_config
    )

    source_dir = tmp_path / "source"
    output_dir = tmp_path / "prepared"

    assert (
        prepare_retrieval_dataset_main(
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


def test_eval_entrypoint_runs_with_hydra_overrides(tmp_path) -> None:
    output_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    assert train_main([f"output_dir={output_dir}", "training/datamodule=synthetic"]) == 0

    assert (
        eval_main(
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


def test_eval_entrypoint_requires_checkpoint_path(tmp_path) -> None:
    _assert_cli_failure(
        lambda: eval_main([f"output_dir={tmp_path / 'eval'}", "training/datamodule=synthetic"]),
        allowed_exceptions=(SystemExit, MissingMandatoryValue, ValueError),
        expected_message="checkpoint_path",
    )


def test_train_entrypoint_refuses_existing_output_dir_without_flags(tmp_path) -> None:
    output_dir = tmp_path / "train"
    assert train_main([f"output_dir={output_dir}", "training/datamodule=synthetic"]) == 0

    _assert_cli_failure(
        lambda: train_main([f"output_dir={output_dir}", "training/datamodule=synthetic"]),
        allowed_exceptions=(SystemExit, FileExistsError),
        expected_message="already contains a saved MedRAP run",
    )


def test_train_entrypoint_supports_resume_from_output_dir(tmp_path) -> None:
    output_dir = tmp_path / "train"
    assert train_main([f"output_dir={output_dir}", "training/datamodule=synthetic"]) == 0

    assert train_main([f"output_dir={output_dir}", "training/datamodule=synthetic", "do_resume=true"]) == 0


def test_eval_entrypoint_supports_test_mode(tmp_path) -> None:
    output_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    assert train_main([f"output_dir={output_dir}", "training/datamodule=synthetic"]) == 0

    assert (
        eval_main(
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
    assert train_main([f"output_dir={output_dir}", "training/datamodule=synthetic"]) == 0
    assert (
        eval_main(
            [
                f"output_dir={eval_dir}",
                f"checkpoint_path={checkpoint_path}",
                "training/datamodule=synthetic",
            ]
        )
        == 0
    )

    _assert_cli_failure(
        lambda: eval_main(
            [
                f"output_dir={eval_dir}",
                f"checkpoint_path={checkpoint_path}",
                "training/datamodule=synthetic",
            ]
        ),
        allowed_exceptions=(SystemExit, FileExistsError),
        expected_message="already contains a saved MedRAP eval run",
    )


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
