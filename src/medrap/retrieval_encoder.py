"""Retrieval encoder modules for mapping retrieved outputs into model memory.

These components consume ``RetrieverOutput`` objects and produce
``RetrievalEncoderOutput`` tensors for fusion.
"""

from torch import nn

from .types import RetrievalEncoderOutput, RetrieverOutput


class TokenFeatureRetrievalEncoder(nn.Module):
    """Minimal sequence-style retrieval encoder that embeds retrieved token ids.

    Args:
        vocab_size: Size of the retrieved document token vocabulary.
        embedding_dim: Size of the per-token feature vector ``D_mem``.
    """

    def __init__(self, *, vocab_size: int = 1024, embedding_dim: int = 4) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

    def encode(self, retrieval: RetrieverOutput) -> RetrievalEncoderOutput:
        """Embed ``retrieval.doc_tokens`` into a dense token-memory tensor.

        Args:
            retrieval: ``RetrieverOutput`` with ``doc_tokens`` shaped
                ``(B, R, K, S_doc)``.

        Returns:
            ``RetrievalEncoderOutput`` where ``retrieval_memory`` has shape
            ``(B, R, K, S_doc, D_mem)``.

        Examples:
            >>> import torch
            >>> retrieval = RetrieverOutput(
            ...     doc_tokens=torch.LongTensor([[[[1, 2, 3]]], [[[4, 5, 6]]]]),
            ...     doc_attention_mask=torch.BoolTensor([[[[True, True, True]]], [[[True, True, True]]]]),
            ... )
            >>> encoder = TokenFeatureRetrievalEncoder(vocab_size=8, embedding_dim=2)
            >>> out = encoder.encode(retrieval)
            >>> tuple(out.retrieval_memory.shape)
            (2, 1, 1, 3, 2)
            >>> out.retrieval_memory.dtype
            torch.float32
        """
        return RetrievalEncoderOutput(retrieval_memory=self.embedding(retrieval.doc_tokens.long()))

    def forward(self, retrieval: RetrieverOutput) -> RetrievalEncoderOutput:
        """Call ``encode``."""
        return self.encode(retrieval)


class MeanPooledRetrievalEncoder(nn.Module):
    """Tabular-style retrieval encoder that mean-pools retrieved token embeddings.

    Args:
        vocab_size: Size of the retrieved document token vocabulary.
        embedding_dim: Output memory dimension ``D_mem``.
    """

    def __init__(self, *, vocab_size: int = 1024, embedding_dim: int = 4) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.token_encoder = TokenFeatureRetrievalEncoder(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
        )

    def encode(self, retrieval: RetrieverOutput) -> RetrievalEncoderOutput:
        """Mask-average token embeddings into a single vector per sample.

        Args:
            retrieval: ``RetrieverOutput`` with token ids and attention mask.

        Returns:
            ``RetrievalEncoderOutput`` where ``retrieval_memory`` has shape
            ``(B, 1, 1, 1, D_mem)``.

        Examples:
            >>> import torch
            >>> retrieval = RetrieverOutput(
            ...     doc_tokens=torch.LongTensor([[[[1, 2, 0]]], [[[4, 0, 0]]]]),
            ...     doc_attention_mask=torch.BoolTensor([[[[True, True, False]]], [[[True, False, False]]]]),
            ... )
            >>> encoder = MeanPooledRetrievalEncoder(vocab_size=8, embedding_dim=2)
            >>> out = encoder.encode(retrieval)
            >>> tuple(out.retrieval_memory.shape)
            (2, 1, 1, 1, 2)
            >>> out.retrieval_memory.dtype
            torch.float32
        """
        token_features = self.token_encoder(retrieval).retrieval_memory
        mask = retrieval.doc_attention_mask.bool().unsqueeze(-1)
        masked_features = token_features * mask
        reduce_dims = tuple(range(1, token_features.ndim - 1))
        counts = mask.sum(dim=reduce_dims).clamp_min(1).to(dtype=token_features.dtype)
        pooled = masked_features.sum(dim=reduce_dims) / counts
        return RetrievalEncoderOutput(retrieval_memory=pooled[:, None, None, None, :])

    def forward(self, retrieval: RetrieverOutput) -> RetrievalEncoderOutput:
        """Call ``encode``."""
        return self.encode(retrieval)
