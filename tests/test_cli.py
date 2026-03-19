from medrap.cli import eval_main, main, prepare_retrieval_dataset_main, train_main

TINY_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
import pytest


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


def test_train_entrypoint_runs_with_hydra_overrides(tmp_path) -> None:
    output_dir = tmp_path / "train"

    assert train_main([f"output_dir={output_dir}", "training/datamodule=synthetic"]) == 0
    assert (output_dir / "checkpoints" / "last.ckpt").exists()


<<<<<<< HEAD
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


def test_eval_entrypoint_requires_checkpoint_path() -> None:
    with pytest.raises(SystemExit):
        eval_main(["output_dir=/tmp/medrap-eval", "training/datamodule=synthetic"])


def test_train_entrypoint_refuses_existing_output_dir_without_flags(tmp_path) -> None:
    output_dir = tmp_path / "train"
    assert train_main([f"output_dir={output_dir}", "training/datamodule=synthetic"]) == 0

    with pytest.raises(SystemExit):
        train_main([f"output_dir={output_dir}", "training/datamodule=synthetic"])


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
