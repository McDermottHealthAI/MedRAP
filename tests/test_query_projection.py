import tempfile
from pathlib import Path

import polars as pl
import torch

from medrap.model.query_projection import Qwen3TextQueryProjector
from medrap.types import EncoderOutput


class _FakeSentenceTransformer:
    """Stand-in for ``sentence_transformers.SentenceTransformer``.

    Encodes each text deterministically by its length, avoiding a real model download in tests.
    """

    def __init__(self, model_name_or_path: str, device: str | None = None) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self._params = [torch.nn.Parameter(torch.zeros(1))]

    def eval(self) -> "_FakeSentenceTransformer":
        return self

    def parameters(self) -> list[torch.nn.Parameter]:
        return self._params

    def encode(self, texts: list[str], *, convert_to_tensor: bool, show_progress_bar: bool) -> torch.Tensor:
        return torch.FloatTensor([[float(len(text))] for text in texts])


def _write_codes_parquet(tmpdir: str) -> Path:
    fp = Path(tmpdir) / "codes.parquet"
    pl.DataFrame(
        {
            "code/vocab_index": [1, 2],
            "code": ["DIAG//A", "DIAG//B"],
            "description": ["Diabetes", "Hypertension"],
        }
    ).write_parquet(fp)
    return fp


def test_qwen3_text_query_projector_embeds_rendered_codes(monkeypatch) -> None:
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer)

    with tempfile.TemporaryDirectory() as tmpdir:
        fp = _write_codes_parquet(tmpdir)
        projector = Qwen3TextQueryProjector(
            model_name_or_path="fake/model",
            code_metadata_path=str(fp),
            max_codes=2,
        )
        for parameter in projector._embedder.parameters():
            assert not parameter.requires_grad

        batch_code = torch.LongTensor([[1, 2, 0], [0, 0, 0]])
        encoder_out = EncoderOutput(patient_state=torch.zeros(2, 1, 1))
        out = projector.project(encoder_out, batch=type("_B", (), {"code": batch_code})())

    assert tuple(out.query_embeddings.shape) == (2, 1, 1)
    assert out.retrieval_step_ids is None
    # "Diabetes, Hypertension" has 22 chars; the all-padding row renders "" (0 chars).
    assert out.query_embeddings[:, 0, 0].tolist() == [22.0, 0.0]


def test_qwen3_text_query_projector_requires_batch(monkeypatch) -> None:
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer)

    with tempfile.TemporaryDirectory() as tmpdir:
        fp = _write_codes_parquet(tmpdir)
        projector = Qwen3TextQueryProjector(model_name_or_path="fake/model", code_metadata_path=str(fp))
        encoder_out = EncoderOutput(patient_state=torch.zeros(1, 1, 1))
        try:
            projector.project(encoder_out, batch=None)
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "requires the raw MEDSTorchBatch" in str(exc)
