from medrap.cli import eval_main, main, prepare_retrieval_dataset_main, train_main


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


def test_prepare_retrieval_dataset_entrypoint_runs_with_hydra_overrides(tmp_path, monkeypatch) -> None:
    from medrap import preparation as prep_module

    source_dir = tmp_path / "source"
    output_dir = tmp_path / "prepared"
    prep_module.Dataset.from_dict({"text": ["alpha", "beta"]}).save_to_disk(str(source_dir))

    monkeypatch.setattr(prep_module, "load_hf_tokenizer", lambda **_kwargs: _StubTokenizer())
    monkeypatch.setattr(prep_module, "load_sentence_transformer", lambda **_kwargs: _StubEmbedder())

    assert (
        prepare_retrieval_dataset_main(
            [
                "prep/source=load_from_disk",
                f"prep.source.dataset_path={source_dir}",
                "prep.document.fields=[text]",
                "prep.tokenizer.model_name=stub-tokenizer",
                "prep.embedder.model_name=stub-embedder",
                f"prep.output.output_dir={output_dir}",
                "prep.index.max_length=3",
            ]
        )
        == 0
    )
    assert (output_dir / "retrieval.faiss").exists()


class _StubTokenizer:
    def __call__(self, texts, *, truncation, padding, max_length):
        return {
            "input_ids": [[idx + 1] * max_length for idx, _ in enumerate(texts)],
            "attention_mask": [[1] * max_length for _ in texts],
        }


class _StubEmbedder:
    def encode(self, texts, *, batch_size, convert_to_numpy, show_progress_bar):
        return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]
