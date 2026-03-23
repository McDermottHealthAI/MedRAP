from medrap.cli import eval_main, main, prepare_retrieval_dataset_main, train_main

TINY_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


def test_medrap_train_cli_runs_with_overrides() -> None:
    assert main(["train"]) == 0


def test_medrap_eval_cli_runs_with_overrides() -> None:
    assert main(["eval"]) == 0


def test_train_entrypoint_runs_with_hydra_overrides() -> None:
    assert train_main([]) == 0


def test_eval_entrypoint_runs_with_hydra_overrides() -> None:
    assert eval_main([]) == 0


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
