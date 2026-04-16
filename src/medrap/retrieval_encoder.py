"""Retrieval encoder modules for mapping retrieved outputs into model memory.

These components consume ``RetrieverOutput`` objects and produce
``RetrievalEncoderOutput`` tensors for fusion.
"""

from torch import nn            
from transformers import AutoModel

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


class LinearProjectionRetrievalEncoder(nn.Module):
    """Project document key embeddings through a learned linear layer.

    Keys (used for FAISS retrieval) and values (fed to fusion) are the same
    source Qwen3 embeddings, but the linear projection gives the model a
    trainable transformation to select which aspects of the contextual
    embedding are useful for the downstream task.

    This keeps the full contextual meaning of ``doc_key_embeddings`` while
    decoupling the value representation from the key representation.

    Args:
        in_dim: Input dimension of ``doc_key_embeddings`` (e.g. 1024 for
            Qwen3-Embedding-0.6B).
        out_dim: Output dimension ``D_mem`` passed to fusion and the head.

    Examples:
        >>> import torch
        >>> from medrap.types import RetrieverOutput
        >>> enc = LinearProjectionRetrievalEncoder(in_dim=4, out_dim=2)
        >>> ro = RetrieverOutput(
        ...     doc_tokens=torch.zeros(2, 1, 3, 1, dtype=torch.long),
        ...     doc_attention_mask=torch.ones(2, 1, 3, 1, dtype=torch.bool),
        ...     doc_key_embeddings=torch.randn(2, 1, 3, 4),
        ... )
        >>> out = enc(ro)
        >>> tuple(out.retrieval_memory.shape)
        (2, 1, 3, 1, 2)
        >>> out.retrieval_memory.dtype
        torch.float32
    """

    def __init__(self, *, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.linear = nn.Linear(self.in_dim, self.out_dim)

    def encode(self, retrieval: RetrieverOutput) -> RetrievalEncoderOutput:
        """Apply linear projection to key embeddings.

        Args:
            retrieval: ``RetrieverOutput`` with ``doc_key_embeddings`` shaped
                ``(B, R, K, D_in)``.

        Returns:
            ``RetrievalEncoderOutput`` where ``retrieval_memory`` has shape
            ``(B, R, K, 1, D_out)``.

        Examples:
            >>> import torch
            >>> from medrap.types import RetrieverOutput
            >>> enc = LinearProjectionRetrievalEncoder(in_dim=4, out_dim=2)
            >>> enc(RetrieverOutput(
            ...     doc_tokens=torch.zeros(1, 1, 1, 1, dtype=torch.long),
            ...     doc_attention_mask=torch.ones(1, 1, 1, 1, dtype=torch.bool),
            ... ))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: ...doc_key_embeddings...
        """
        keys = retrieval.doc_key_embeddings
        if keys is None:
            raise ValueError("LinearProjectionRetrievalEncoder requires doc_key_embeddings on RetrieverOutput")
        projected = self.linear(keys.float())   # (B, R, K, D_out)
        return RetrievalEncoderOutput(retrieval_memory=projected.unsqueeze(3))  # (B, R, K, 1, D_out)

    def forward(self, retrieval: RetrieverOutput) -> RetrievalEncoderOutput:
        """Call ``encode``."""
        return self.encode(retrieval)


class KeyEmbeddingRetrievalEncoder(nn.Module):
    """Use stored document key embeddings as per-document memory.

    Packs ``RetrieverOutput.doc_key_embeddings`` into ``retrieval_memory`` with
    shape ``(B, R, K, 1, D)`` so :class:`medrap.fusion.ReplaceFusion` can emit a
    per-document fused state ``(B, K, D)`` when ``R = 1`` and ``K >= 1``.

    Requires ``doc_key_embeddings`` on the retriever output.
    """

    def __init__(self) -> None:
        super().__init__()

    def encode(self, retrieval: RetrieverOutput) -> RetrievalEncoderOutput:
        """View key embeddings as singleton-length document memory.

        Examples:
            >>> import torch
            >>> from medrap.types import RetrieverOutput
            >>> enc = KeyEmbeddingRetrievalEncoder()
            >>> ro = RetrieverOutput(
            ...     doc_tokens=torch.zeros(1, 1, 1, 1, dtype=torch.long),
            ...     doc_attention_mask=torch.ones(1, 1, 1, 1, dtype=torch.bool),
            ...     doc_key_embeddings=None,
            ... )
            >>> enc(ro)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: ...doc_key_embeddings...
        """
        keys = retrieval.doc_key_embeddings
        if keys is None:
            raise ValueError("KeyEmbeddingRetrievalEncoder requires doc_key_embeddings on RetrieverOutput")
        return RetrievalEncoderOutput(retrieval_memory=keys.float().unsqueeze(3))

    def forward(self, retrieval: RetrieverOutput) -> RetrievalEncoderOutput:
        """Call ``encode``."""
        return self.encode(retrieval)


class PerDocMeanPooledRetrievalEncoder(nn.Module):
    """Per-document mean-pooled retrieval encoder using learned token embeddings.

    Unlike :class:`MeanPooledRetrievalEncoder` (which collapses all K docs into
    one global vector), this encoder pools over the ``S_doc`` token dimension
    only, preserving the ``K`` document dimension.  The output shape
    ``(B, R, K, 1, D_mem)`` is compatible with
    :class:`medrap.fusion.ReplaceFusion` and the marginalized retrieval path,
    where keys (Qwen3 embeddings used for FAISS retrieval) and values (the
    learned embeddings produced here) are different representations.

    Args:
        vocab_size: Token vocabulary size.
        embedding_dim: Output embedding dimension ``D_mem``.
        pretrained_model_name_or_path: Optional HuggingFace model name or local
            path.  When provided, the token embedding matrix is copied from
            ``model.embed_tokens`` of the pretrained model and used to
            initialise ``self.embedding``.  The pretrained model is loaded
            in full precision, weights are copied, then the model is deleted to
            free memory.  ``vocab_size`` and ``embedding_dim`` must match the
            pretrained embedding shape exactly.

    Examples:
        >>> import torch
        >>> from medrap.types import RetrieverOutput
        >>> enc = PerDocMeanPooledRetrievalEncoder(vocab_size=8, embedding_dim=4)
        >>> ro = RetrieverOutput(
        ...     doc_tokens=torch.LongTensor([[[[1, 2, 0], [3, 0, 0]]]]),
        ...     doc_attention_mask=torch.BoolTensor([[[[True, True, False], [True, False, False]]]]),
        ... )
        >>> out = enc(ro)
        >>> tuple(out.retrieval_memory.shape)
        (1, 1, 2, 1, 4)
        >>> out.retrieval_memory.dtype
        torch.float32
    """

    def __init__(
        self,
        *,
        vocab_size: int = 1024,
        embedding_dim: int = 4,
        pretrained_model_name_or_path: str | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        if pretrained_model_name_or_path is not None:
            pretrained = AutoModel.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
            )
            embed_tokens = getattr(pretrained, "embed_tokens", None) or getattr(pretrained.model, "embed_tokens")
            pretrained_weight = embed_tokens.weight.detach().float()
            pre_v, pre_d = pretrained_weight.shape
            if pre_v != self.vocab_size or pre_d != self.embedding_dim:
                raise ValueError(
                    f"Pretrained embedding shape ({pre_v}, {pre_d}) does not match "
                    f"(vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}). "
                    f"Use vocab_size={pre_v} embedding_dim={pre_d} to match this model."
                )
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained_weight)
            del pretrained

    def encode(self, retrieval: RetrieverOutput) -> RetrievalEncoderOutput:
        """Embed tokens and mean-pool over S_doc, keeping the K doc dimension.

        Args:
            retrieval: ``RetrieverOutput`` with ``doc_tokens`` shaped
                ``(B, R, K, S_doc)`` and ``doc_attention_mask`` shaped
                ``(B, R, K, S_doc)``.

        Returns:
            ``RetrievalEncoderOutput`` where ``retrieval_memory`` has shape
            ``(B, R, K, 1, D_mem)``.

        Examples:
            >>> import torch
            >>> from medrap.types import RetrieverOutput
            >>> enc = PerDocMeanPooledRetrievalEncoder(vocab_size=8, embedding_dim=4)

            Masked tokens do not contribute — a doc with one valid token plus a
            masked token yields the same vector as a doc with only that token:

            >>> ro_one = RetrieverOutput(
            ...     doc_tokens=torch.LongTensor([[[[1, 0]]]]),
            ...     doc_attention_mask=torch.BoolTensor([[[[True, False]]]]),
            ... )
            >>> ro_two = RetrieverOutput(
            ...     doc_tokens=torch.LongTensor([[[[1, 7]]]]),
            ...     doc_attention_mask=torch.BoolTensor([[[[True, False]]]]),
            ... )
            >>> (enc(ro_one).retrieval_memory == enc(ro_two).retrieval_memory).all().item()
            True

            A fully-masked document produces a zero vector (clamp_min(1) keeps
            the denominator safe; the masked sum is zero):

            >>> ro_all_masked = RetrieverOutput(
            ...     doc_tokens=torch.LongTensor([[[[1, 2]]]]),
            ...     doc_attention_mask=torch.BoolTensor([[[[False, False]]]]),
            ... )
            >>> enc(ro_all_masked).retrieval_memory.sum().item()
            0.0
        """
        token_features = self.embedding(retrieval.doc_tokens.long())   # (B, R, K, S_doc, D_mem)
        mask = retrieval.doc_attention_mask.bool().unsqueeze(-1)        # (B, R, K, S_doc, 1)
        masked = token_features * mask
        counts = mask.sum(dim=3).clamp_min(1).to(dtype=token_features.dtype)  # (B, R, K, 1)
        pooled = masked.sum(dim=3) / counts                             # (B, R, K, D_mem)
        return RetrievalEncoderOutput(retrieval_memory=pooled.unsqueeze(3))    # (B, R, K, 1, D_mem)

    def forward(self, retrieval: RetrieverOutput) -> RetrievalEncoderOutput:
        """Call ``encode``."""
        return self.encode(retrieval)
