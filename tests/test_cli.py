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


def test_prepare_retrieval_dataset_entrypoint_runs_with_hydra_overrides(tmp_path) -> None:
    from medrap import preparation as prep_module

    source_dir = tmp_path / "source"
    output_dir = tmp_path / "prepared"
    prep_module.Dataset.from_dict({"text": ["alpha", "beta"]}).save_to_disk(str(source_dir))

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
    assert (output_dir / "retrieval.faiss").exists()
