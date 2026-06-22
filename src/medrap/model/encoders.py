"""Patient-side encoder modules for retrieval-augmented modeling.

These components convert MEDS batch inputs into dense patient representation used by query projection and
downstream fusion.
"""

import logging
from abc import ABC, abstractmethod

import torch
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn
from torch.nn import functional as nn_functional

from ..types import EncoderOutput

log = logging.getLogger(__name__)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embeddings to x.

    Args:
        x: ``(B, S, H, D_h)``
        cos: ``(B, S, 1, D_h)`` — broadcasts over heads
        sin: ``(B, S, 1, D_h)`` — broadcasts over heads

    Returns:
        Rotated tensor with the same shape as ``x``.

    Examples:
        >>> import torch
        >>> x = torch.randn(2, 3, 4, 8)
        >>> cos = torch.ones(2, 3, 1, 8)
        >>> sin = torch.zeros(2, 3, 1, 8)
        >>> out = _apply_rope(x, cos, sin)
        >>> tuple(out.shape)
        (2, 3, 4, 8)
        >>> torch.allclose(out, x)
        True
    """
    x_rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    return x * cos + x_rot * sin


def _time_delta_rope_freqs(
    time_delta_days: Tensor,
    head_dim: int,
    padding_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute RoPE cos/sin from cumulative log-time deltas.

    Zeroes out time deltas at padding positions before the cumulative sum so
    padding tokens do not shift the time axis.

    Args:
        time_delta_days: ``(B, S)`` time gaps in days between consecutive events.
        head_dim: Size of a single attention head (must be even).
        padding_mask: ``(B, S)`` bool tensor, ``True`` where ``code == 0``.

    Returns:
        Tuple of ``(cos, sin)``, each shaped ``(B, S, 1, head_dim)``.

    Examples:
        >>> import torch
        >>> t = torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 0.0]])
        >>> mask = torch.zeros(2, 3, dtype=torch.bool)
        >>> cos, sin = _time_delta_rope_freqs(t, 8, mask)
        >>> tuple(cos.shape)
        (2, 3, 1, 8)
        >>> tuple(sin.shape)
        (2, 3, 1, 8)

        At ``t=0`` the rotation is the identity (``cos=1``, ``sin=0``):

        >>> mask2 = torch.tensor([[False, False, True]])
        >>> cos2, sin2 = _time_delta_rope_freqs(torch.zeros(1, 3), 4, mask2)
        >>> torch.allclose(cos2[0, 0], torch.ones(1, 4))
        True
        >>> torch.allclose(sin2[0, 0], torch.zeros(1, 4), atol=1e-6)
        True

        A padding token (mask=True) does not advance the time axis for the
        positions that follow it:

        >>> padded = torch.tensor([[0.0, 1.0, 5.0]])
        >>> pad_mask = torch.tensor([[False, False, True]])
        >>> cos_padded, _ = _time_delta_rope_freqs(padded, 4, pad_mask)
        >>> cos_no_pad, _ = _time_delta_rope_freqs(
        ...     torch.tensor([[0.0, 1.0, 0.0]]), 4, torch.zeros(1, 3, dtype=torch.bool)
        ... )
        >>> torch.allclose(cos_padded[:, :2], cos_no_pad[:, :2])
        True

        Negative time deltas (a data-quality issue) are clamped to 0 rather
        than raising, matching a zero-gap event at that position:

        >>> negative = torch.tensor([[0.0, -1.0, 2.0]])
        >>> cos_neg, _ = _time_delta_rope_freqs(negative, 4, torch.zeros(1, 3, dtype=torch.bool))
        >>> zero_gap, _ = _time_delta_rope_freqs(
        ...     torch.tensor([[0.0, 0.0, 2.0]]), 4, torch.zeros(1, 3, dtype=torch.bool)
        ... )
        >>> torch.allclose(cos_neg, zero_gap)
        True
    """
    if (time_delta_days < 0).any():
        log.debug("_time_delta_rope_freqs received negative time deltas; clamping to 0.")
    td = time_delta_days.float().masked_fill(padding_mask, 0.0)
    t = torch.cumsum(torch.log1p(td.clamp(min=0)), dim=1)  # (B, S)
    freqs = 10000.0 ** (
        -torch.arange(head_dim // 2, device=td.device, dtype=td.dtype) / (head_dim // 2)
    )  # (head_dim//2,)
    angles = t.unsqueeze(-1) * freqs  # (B, S, head_dim//2)
    cos = angles.cos().repeat_interleave(2, dim=-1).unsqueeze(2)  # (B, S, 1, head_dim)
    sin = angles.sin().repeat_interleave(2, dim=-1).unsqueeze(2)
    return cos, sin


class _TimeDeltaRoPEAttention(nn.Module):
    """Multi-head self-attention with time-delta RoPE applied to Q and K."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = float(dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, seq_len, _ = x.shape
        n_heads, head_dim = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(batch_size, seq_len, n_heads, head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, n_heads, head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, n_heads, head_dim)

        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)

        q = q.transpose(1, 2)  # (B, H, S, D_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask: Tensor | None = None
        if key_padding_mask is not None:
            # scaled_dot_product_attention's bool mask convention is the opposite of
            # key_padding_mask: True means "attend", not "this is padding".
            attn_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)

        out = nn_functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(batch_size, seq_len, n_heads * head_dim)
        return self.out_proj(out)


class _TimeDeltaRoPELayer(nn.Module):
    """Single transformer layer: LN → RoPE attention → residual → LN → GELU FFN → residual."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = _TimeDeltaRoPEAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        attn_out = self.attn(self.norm1(x), rope_cos, rope_sin, key_padding_mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class PatientEncoder(nn.Module, ABC):
    """Abstract base for all patient encoders.

    Subclasses must implement :meth:`encode`, which maps a ``MEDSTorchBatch``
    to an ``EncoderOutput``.  The ``forward`` method delegates to ``encode``
    so that the encoder can be used as a standard ``nn.Module``.
    """

    @abstractmethod
    def encode(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Encode a patient batch into a dense representation.

        Args:
            batch: A ``MEDSTorchBatch``.

        Returns:
            An ``EncoderOutput`` whose ``patient_state`` has shape
            ``(B, S_ehr, D_ehr)``.
        """

    def forward(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Call ``encode``."""
        return self.encode(batch)


class MEDSCodeEncoder(PatientEncoder):
    """Scaffold sequence encoder that casts ``batch.code`` to a float representation.

    This is a minimal, non-learned encoder for MEDS-style batches. It converts
    the integer code ids to floats and unsqueezes a trailing dimension so the
    output satisfies the sequence-mode shape contract ``(B, S_ehr, D_ehr)``
    with ``D_ehr = 1``.
    """

    def __init__(self) -> None:
        super().__init__()

    def encode(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Return ``batch.code`` as a float tensor with a trailing embedding dim.

        Args:
            batch: A ``MEDSTorchBatch`` containing a ``code`` field of shape
                ``(B, S_ehr)``.

        Returns:
            An ``EncoderOutput`` where ``patient_state`` has shape
            ``(B, S_ehr, 1)`` and ``attention_mask`` (``True`` = valid,
            non-padding position) has shape ``(B, S_ehr)``.

        Examples:
            >>> encoder = MEDSCodeEncoder()
            >>> batch = MEDSTorchBatch(
            ...     code=torch.LongTensor([[11, 22, 0], [7, 3, 1]]),
            ...     numeric_value=torch.zeros(2, 3),
            ...     numeric_value_mask=torch.zeros(2, 3, dtype=torch.bool),
            ...     time_delta_days=torch.zeros(2, 3),
            ... )
            >>> out = encoder.encode(batch)
            >>> tuple(out.patient_state.shape)
            (2, 3, 1)
            >>> out.patient_state.dtype
            torch.float32
            >>> out.attention_mask.tolist()
            [[True, True, False], [True, True, True]]
        """
        return EncoderOutput(patient_state=batch.code.float().unsqueeze(-1), attention_mask=batch.code != 0)


class TokenEmbeddingEncoder(PatientEncoder):
    """Sequence encoder that maps ``batch.code`` to learned token embeddings.

    This is a minimal learned sequence encoder for MEDS-style batches. It reads
    ``batch.code`` from a ``MEDSTorchBatch`` and maps each code id to an
    embedding vector, producing a dense patient representation. The embedding
    table uses ``padding_idx=0`` so padding tokens (``code == 0``) embed to zero.

    Args:
        vocab_size: Size of the EHR code vocabulary.
        embedding_dim: Output hidden size ``D_ehr``.
    """

    def __init__(self, *, vocab_size: int = 1024, embedding_dim: int = 4) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)

    def encode(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Embed ``batch.code`` into a sequence hidden state.

        Args:
            batch: A ``MEDSTorchBatch`` containing a ``code`` field of shape
                ``(B, S_ehr)``.

        Returns:
            An ``EncoderOutput`` where ``patient_state`` has shape
            ``(B, S_ehr, D_ehr)`` and ``attention_mask`` (``True`` = valid,
            non-padding position) has shape ``(B, S_ehr)``.

        Examples:
            >>> encoder = TokenEmbeddingEncoder(vocab_size=8, embedding_dim=2)
            >>> batch = MEDSTorchBatch(
            ...     code=torch.LongTensor([[1, 2, 0], [3, 4, 5]]),
            ...     numeric_value=torch.zeros(2, 3),
            ...     numeric_value_mask=torch.zeros(2, 3, dtype=torch.bool),
            ...     time_delta_days=torch.zeros(2, 3),
            ... )
            >>> out = encoder.encode(batch)
            >>> tuple(out.patient_state.shape)
            (2, 3, 2)
            >>> out.patient_state.dtype
            torch.float32
            >>> out.attention_mask.tolist()
            [[True, True, False], [True, True, True]]

            Padding (``code == 0``) embeds to zero:

            >>> bool((out.patient_state[0, 2] == 0).all())
            True
        """
        code = batch.code.long()
        return EncoderOutput(patient_state=self.embedding(code), attention_mask=code != 0)


class TimeDeltaRoPEPatientEncoder(PatientEncoder):
    """Transformer encoder with time-delta rotary position embeddings (RoPE).

    Each EHR code is first mapped to a learned embedding, then multi-head
    pre-norm self-attention layers enrich the sequence using rotary position
    embeddings derived from cumulative log-time deltas between events. Padding positions
    (``code == 0``) are masked from self-attention, and the embedding table uses
    ``padding_idx=0`` so padding tokens remain zero-initialised.

    Args:
        vocab_size: Size of the EHR code vocabulary.
        embedding_dim: Token embedding and transformer hidden size ``D_ehr``.
            Must be divisible by ``num_heads``.
        num_heads: Number of attention heads.
        num_layers: Number of stacked transformer layers.
        ff_dim: Inner size of the position-wise feed-forward network.
        dropout: Dropout probability applied inside attention and FFN.

    Examples:
        >>> TimeDeltaRoPEPatientEncoder(vocab_size=16, embedding_dim=9, num_heads=4)
        Traceback (most recent call last):
            ...
        ValueError: embedding_dim=9 must be divisible by num_heads=4
    """

    def __init__(
        self,
        *,
        vocab_size: int = 65536,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(embedding_dim)
        h = int(num_heads)
        if d % h != 0:
            raise ValueError(f"embedding_dim={d} must be divisible by num_heads={h}")
        self._head_dim = d // h
        self.embedding = nn.Embedding(int(vocab_size), d, padding_idx=0)
        self.layers = nn.ModuleList(
            [_TimeDeltaRoPELayer(d, h, int(ff_dim), float(dropout)) for _ in range(int(num_layers))]
        )
        self.norm = nn.LayerNorm(d)

    def encode(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Encode patient codes with time-aware rotary self-attention.

        Args:
            batch: A ``MEDSTorchBatch`` with ``code`` ``(B, S_ehr)`` and
                ``time_delta_days`` ``(B, S_ehr)``.

        Returns:
            An ``EncoderOutput`` where ``patient_state`` has shape
            ``(B, S_ehr, D_ehr)`` and ``attention_mask`` (``True`` = valid,
            non-padding position) has shape ``(B, S_ehr)``.

        Examples:
            >>> import torch
            >>> from meds_torchdata import MEDSTorchBatch
            >>> encoder = TimeDeltaRoPEPatientEncoder(
            ...     vocab_size=16, embedding_dim=8, num_heads=2, num_layers=1, ff_dim=16
            ... )
            >>> batch = MEDSTorchBatch(
            ...     code=torch.LongTensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            ...     numeric_value=torch.zeros(2, 4),
            ...     numeric_value_mask=torch.zeros(2, 4, dtype=torch.bool),
            ...     time_delta_days=torch.tensor([[0.0, 1.0, 2.0, 0.0], [0.0, 3.0, 0.0, 0.0]]),
            ... )
            >>> out = encoder.encode(batch)
            >>> tuple(out.patient_state.shape)
            (2, 4, 8)
            >>> out.patient_state.dtype
            torch.float32
            >>> tuple(encoder(batch).patient_state.shape)
            (2, 4, 8)
            >>> out.attention_mask.tolist()
            [[True, True, True, False], [True, True, False, False]]

            A single-token batch is encoded without error:

            >>> single = MEDSTorchBatch(
            ...     code=torch.LongTensor([[1]]),
            ...     numeric_value=torch.zeros(1, 1),
            ...     numeric_value_mask=torch.zeros(1, 1, dtype=torch.bool),
            ...     time_delta_days=torch.zeros(1, 1),
            ... )
            >>> encoder.encode(single).patient_state.shape
            torch.Size([1, 1, 8])

            An all-padding batch does not raise:

            >>> all_padding = MEDSTorchBatch(
            ...     code=torch.LongTensor([[0, 0, 0]]),
            ...     numeric_value=torch.zeros(1, 3),
            ...     numeric_value_mask=torch.zeros(1, 3, dtype=torch.bool),
            ...     time_delta_days=torch.zeros(1, 3),
            ... )
            >>> encoder.encode(all_padding).patient_state.shape
            torch.Size([1, 3, 8])

            Trailing padding does not change the encoding of valid tokens:

            >>> _ = encoder.eval()
            >>> short = MEDSTorchBatch(
            ...     code=torch.LongTensor([[1, 2, 3]]),
            ...     numeric_value=torch.zeros(1, 3),
            ...     numeric_value_mask=torch.zeros(1, 3, dtype=torch.bool),
            ...     time_delta_days=torch.tensor([[0.0, 1.0, 2.0]]),
            ... )
            >>> padded = MEDSTorchBatch(
            ...     code=torch.LongTensor([[1, 2, 3, 0, 0]]),
            ...     numeric_value=torch.zeros(1, 5),
            ...     numeric_value_mask=torch.zeros(1, 5, dtype=torch.bool),
            ...     time_delta_days=torch.tensor([[0.0, 1.0, 2.0, 0.0, 0.0]]),
            ... )
            >>> with torch.no_grad():
            ...     out_short = encoder.encode(short).patient_state[0, :3]
            ...     out_padded = encoder.encode(padded).patient_state[0, :3]
            >>> torch.allclose(out_short, out_padded, atol=1e-5)
            True
        """
        code = batch.code.long()
        padding_mask = code == 0  # (B, S) True = padding
        rope_cos, rope_sin = _time_delta_rope_freqs(batch.time_delta_days, self._head_dim, padding_mask)
        x = self.embedding(code)  # (B, S, D)
        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin, padding_mask)
        return EncoderOutput(patient_state=self.norm(x), attention_mask=~padding_mask)
