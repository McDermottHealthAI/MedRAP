"""Tests for TimeDeltaRoPEPatientEncoder and its helpers."""

import pytest
import torch
from meds_torchdata import MEDSTorchBatch

from medrap.model.encoders import TimeDeltaRoPEPatientEncoder, _apply_rope, _time_delta_rope_freqs


def _make_batch(
    codes: list[list[int]],
    time_deltas: list[list[float]],
) -> MEDSTorchBatch:
    return MEDSTorchBatch(
        code=torch.LongTensor(codes),
        numeric_value=torch.zeros(len(codes), len(codes[0])),
        numeric_value_mask=torch.zeros(len(codes), len(codes[0]), dtype=torch.bool),
        time_delta_days=torch.tensor(time_deltas, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# _apply_rope
# ---------------------------------------------------------------------------


def test_apply_rope_identity_when_sin_zero() -> None:
    x = torch.randn(2, 3, 4, 8)
    cos = torch.ones(2, 3, 1, 8)
    sin = torch.zeros(2, 3, 1, 8)
    assert torch.allclose(_apply_rope(x, cos, sin), x)


def test_apply_rope_preserves_shape() -> None:
    x = torch.randn(2, 5, 4, 8)
    cos = torch.randn(2, 5, 1, 8)
    sin = torch.randn(2, 5, 1, 8)
    assert _apply_rope(x, cos, sin).shape == x.shape


# ---------------------------------------------------------------------------
# _time_delta_rope_freqs
# ---------------------------------------------------------------------------


def test_time_delta_rope_freqs_output_shape() -> None:
    cos, sin = _time_delta_rope_freqs(torch.zeros(2, 6), 8, torch.zeros(2, 6, dtype=torch.bool))
    assert cos.shape == (2, 6, 1, 8)
    assert sin.shape == (2, 6, 1, 8)


def test_time_delta_rope_freqs_zero_time_gives_identity() -> None:
    """At t=0, cos=1 and sin=0 → rotation is identity."""
    cos, sin = _time_delta_rope_freqs(torch.zeros(1, 3), 4, torch.zeros(1, 3, dtype=torch.bool))
    assert torch.allclose(cos[0, 0], torch.ones(1, 4))
    assert torch.allclose(sin[0, 0], torch.zeros(1, 4), atol=1e-6)


def test_time_delta_rope_freqs_padding_does_not_advance_time() -> None:
    """A padding token (mask=True) should not shift the time axis."""
    t = torch.tensor([[0.0, 1.0, 5.0]])
    mask = torch.tensor([[False, False, True]])
    cos_masked, _ = _time_delta_rope_freqs(t, 4, mask)

    cos_no_pad, _ = _time_delta_rope_freqs(
        torch.tensor([[0.0, 1.0, 0.0]]), 4, torch.zeros(1, 3, dtype=torch.bool)
    )
    assert torch.allclose(cos_masked[:, :2], cos_no_pad[:, :2])


# ---------------------------------------------------------------------------
# TimeDeltaRoPEPatientEncoder
# ---------------------------------------------------------------------------


def test_encoder_output_shape() -> None:
    enc = TimeDeltaRoPEPatientEncoder(vocab_size=16, embedding_dim=8, num_heads=2, num_layers=1, ff_dim=16)
    out = enc(_make_batch([[1, 2, 3, 0], [4, 5, 0, 0]], [[0.0, 1.0, 2.0, 0.0], [0.0, 3.0, 0.0, 0.0]]))
    assert out.patient_state.shape == (2, 4, 8)
    assert out.patient_state.dtype == torch.float32


def test_encoder_forward_equals_encode() -> None:
    enc = TimeDeltaRoPEPatientEncoder(vocab_size=16, embedding_dim=8, num_heads=2, num_layers=1, ff_dim=16)
    batch = _make_batch([[1, 2, 3], [4, 5, 6]], [[0.0, 1.0, 2.0]] * 2)
    assert torch.equal(enc.encode(batch).patient_state, enc(batch).patient_state)


def test_encoder_padding_does_not_affect_valid_tokens() -> None:
    """Valid token outputs should be identical regardless of trailing padding."""
    enc = TimeDeltaRoPEPatientEncoder(vocab_size=16, embedding_dim=8, num_heads=2, num_layers=1, ff_dim=16)
    enc.eval()
    with torch.no_grad():
        out_short = enc(_make_batch([[1, 2, 3]], [[0.0, 1.0, 2.0]])).patient_state[0, :3]
        out_padded = enc(_make_batch([[1, 2, 3, 0, 0]], [[0.0, 1.0, 2.0, 0.0, 0.0]])).patient_state[0, :3]
    assert torch.allclose(out_short, out_padded, atol=1e-5)


def test_encoder_invalid_head_split_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TimeDeltaRoPEPatientEncoder(vocab_size=16, embedding_dim=9, num_heads=4)


def test_encoder_single_token() -> None:
    enc = TimeDeltaRoPEPatientEncoder(vocab_size=16, embedding_dim=8, num_heads=2, num_layers=1, ff_dim=16)
    assert enc(_make_batch([[1]], [[0.0]])).patient_state.shape == (1, 1, 8)


def test_encoder_all_padding_does_not_raise() -> None:
    enc = TimeDeltaRoPEPatientEncoder(vocab_size=16, embedding_dim=8, num_heads=2, num_layers=1, ff_dim=16)
    assert enc(_make_batch([[0, 0, 0]], [[0.0, 0.0, 0.0]])).patient_state.shape == (1, 3, 8)
