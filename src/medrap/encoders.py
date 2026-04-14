"""Patient-side encoder modules for retrieval-augmented modeling.

These components convert MEDS batch inputs into dense patient representation used by query projection and
downstream fusion.
"""

import math
from abc import ABC, abstractmethod

import torch
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn

from .types import EncoderOutput


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
            ``(B, S_ehr, 1)``.

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
        """
        return EncoderOutput(patient_state=batch.code.float().unsqueeze(-1))


class TokenEmbeddingEncoder(PatientEncoder):
    """Sequence encoder that maps ``batch.code`` to learned token embeddings.

    This is a minimal learned sequence encoder for MEDS-style batches. It reads
    ``batch.code`` from a ``MEDSTorchBatch`` and maps each code id to an
    embedding vector, producing a dense patient representation.

    Args:
        vocab_size: Size of the EHR code vocabulary.
        embedding_dim: Output hidden size ``D_ehr``.
    """

    def __init__(self, *, vocab_size: int = 1024, embedding_dim: int = 4) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

    def encode(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Embed ``batch.code`` into a sequence hidden state.

        Args:
            batch: A ``MEDSTorchBatch`` containing a ``code`` field of shape
                ``(B, S_ehr)``.

        Returns:
            An ``EncoderOutput`` where ``patient_state`` has shape
            ``(B, S_ehr, D_ehr)``.

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
        """
        return EncoderOutput(patient_state=self.embedding(batch.code.long()))


class TabularEncoder(PatientEncoder):
    """Tabular encoder that pools a code sequence into a single patient vector.

    Embeds ``batch.code`` via a learned embedding table and mean-pools across the
    sequence dimension to produce a ``(B, 1, D_ehr)`` patient representation.

    Args:
        vocab_size: Size of the EHR code vocabulary.
        embedding_dim: Output hidden size ``D_ehr``.
    """

    def __init__(self, *, vocab_size: int = 1024, embedding_dim: int = 4) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

    def encode(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Embed and mean-pool ``batch.code`` into a tabular patient state.

        Args:
            batch: A ``MEDSTorchBatch`` containing a ``code`` field of shape
                ``(B, S_ehr)``.

        Returns:
            An ``EncoderOutput`` where ``patient_state`` has shape
            ``(B, 1, D_ehr)``.

        Examples:
            >>> encoder = TabularEncoder(vocab_size=8, embedding_dim=2)
            >>> batch = MEDSTorchBatch(
            ...     code=torch.LongTensor([[1, 2, 0], [3, 4, 5]]),
            ...     numeric_value=torch.zeros(2, 3),
            ...     numeric_value_mask=torch.zeros(2, 3, dtype=torch.bool),
            ...     time_delta_days=torch.zeros(2, 3),
            ... )
            >>> out = encoder.encode(batch)
            >>> tuple(out.patient_state.shape)
            (2, 1, 2)
            >>> out.patient_state.dtype
            torch.float32
            >>> tuple(encoder(batch).patient_state.shape)
            (2, 1, 2)
        """
        embedded = self.embedding(batch.code.long())  # (B, S_ehr, D_ehr)
        pooled = embedded.mean(dim=1, keepdim=True)  # (B, 1, D_ehr)
        return EncoderOutput(patient_state=pooled)


class TransformerPatientEncoder(PatientEncoder):
    """Transformer encoder that produces contextualised per-token patient representations.

    Each EHR code is first mapped to a learned embedding, then sinusoidal positional
    encodings are added, and the sequence is passed through a stack of
    ``nn.TransformerEncoderLayer`` modules (pre-norm, ``batch_first=True``).
    Padding positions (``code == 0``) are masked out so they do not contribute to
    self-attention.  The output retains the full ``(B, S_ehr, D_ehr)`` shape so it
    is a drop-in replacement for ``TokenEmbeddingEncoder`` in any downstream
    query projector.

    Args:
        vocab_size: Size of the EHR code vocabulary.
        embedding_dim: Token embedding and transformer hidden size ``D_ehr``.
            Must be divisible by ``num_heads``.
        num_heads: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        feedforward_dim: Inner size of the position-wise FFN.
            Defaults to ``4 * embedding_dim``.
        dropout: Dropout probability applied inside each encoder layer.
        max_seq_len: Maximum sequence length for positional encoding buffer.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 1024,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        ff_dim = int(feedforward_dim) if feedforward_dim is not None else 4 * self.embedding_dim
        # padding_idx=0 keeps the padding embedding fixed at zero
        self.embedding = nn.Embedding(int(vocab_size), self.embedding_dim, padding_idx=0)
        self.register_buffer("pe", self._sinusoidal_pe(int(max_seq_len), self.embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=int(num_heads),
            dim_feedforward=ff_dim,
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,  # pre-norm: more stable when training from scratch
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=int(num_layers), enable_nested_tensor=False
        )

    @staticmethod
    def _sinusoidal_pe(max_seq_len: int, d_model: int) -> Tensor:
        position = torch.arange(max_seq_len).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def encode(self, batch: MEDSTorchBatch) -> EncoderOutput:
        """Encode a patient code sequence with self-attention.

        Args:
            batch: A ``MEDSTorchBatch`` containing a ``code`` field of shape
                ``(B, S_ehr)``.

        Returns:
            An ``EncoderOutput`` where ``patient_state`` has shape
            ``(B, S_ehr, D_ehr)`` with contextualised token representations.

        Examples:
            >>> encoder = TransformerPatientEncoder(
            ...     vocab_size=16, embedding_dim=8, num_heads=2, num_layers=1
            ... )
            >>> batch = MEDSTorchBatch(
            ...     code=torch.LongTensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            ...     numeric_value=torch.zeros(2, 4),
            ...     numeric_value_mask=torch.zeros(2, 4, dtype=torch.bool),
            ...     time_delta_days=torch.zeros(2, 4),
            ... )
            >>> out = encoder.encode(batch)
            >>> tuple(out.patient_state.shape)
            (2, 4, 8)
            >>> out.patient_state.dtype
            torch.float32
            >>> # padding positions (code==0) should be near-zero after masking
            >>> out.patient_state[1, 2, :].abs().max().item() < 0.1
            True
        """
        code = batch.code.long()  # (B, S)
        padding_mask = code == 0  # (B, S) — True marks positions to ignore
        x = self.embedding(code)  # (B, S, D)
        x = x + self.pe[:, : x.size(1), :]  # sinusoidal positions
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # (B, S, D)
        return EncoderOutput(patient_state=x)
