"""Fusion modules for combining patient and retrieval representations.

These components define how retrieved memory contributes to downstream model states before pooling and
prediction.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn

from .types import FusionInput, FusionOutput


class FusionModule(nn.Module, ABC):
    """Abstract base for all fusion modules.

    Subclasses must implement :meth:`fuse`, which combines patient state and
    retrieval memory into a fused representation.  The ``forward`` method
    delegates to ``fuse`` so that the module can be used as a standard
    ``nn.Module``.
    """

    @abstractmethod
    def fuse(self, fusion_input: FusionInput) -> FusionOutput:
        """Fuse patient state with retrieval memory.

        Args:
            fusion_input: A ``FusionInput`` payload.

        Returns:
            A ``FusionOutput`` whose ``fused_state`` has shape
            ``(B, S_ehr, D_fused)``.
        """

    def forward(self, fusion_input: FusionInput) -> FusionOutput:
        """Call ``fuse``."""
        return self.fuse(fusion_input)


class ReplaceFusion(FusionModule):
    """Tabular fusion that discards patient state and uses retrieval memory.

    Supported ``retrieval_memory`` layouts:

    - Legacy pooled memory ``(B, 1, 1, 1, D_mem)`` -> ``fused_state`` ``(B, 1, D_mem)``.
    - Per-document keys ``(B, 1, K, 1, D_mem)`` (or ``S_doc > 1``) -> ``fused_state``
      ``(B, K, D_mem)`` when ``R = 1`` and the singleton doc-length dim is 1; otherwise
      ``(B, R * K * S_doc, D_mem)``.
    """

    def __init__(self) -> None:
        super().__init__()

    def fuse(self, fusion_input: FusionInput) -> FusionOutput:
        """Return retrieval memory as fused patient-side sequence features.

        Args:
            fusion_input: ``FusionInput`` with 5D ``retrieval_memory``.

        Returns:
            ``FusionOutput`` with ``fused_state`` shaped ``(B, S, D_mem)``.

        Examples:
            >>> from medrap.types import FusionInput
            >>> fusion = ReplaceFusion()
            >>> fusion_input = FusionInput(
            ...     patient_state=torch.randn(2, 1, 3),
            ...     retrieval_memory=torch.randn(2, 1, 1, 1, 4),
            ... )
            >>> out = fusion.fuse(fusion_input)
            >>> tuple(out.fused_state.shape)
            (2, 1, 4)
            >>> tuple(fusion(fusion_input).fused_state.shape)
            (2, 1, 4)
            >>> per_doc = FusionInput(
            ...     patient_state=torch.randn(2, 1, 3),
            ...     retrieval_memory=torch.randn(2, 1, 3, 1, 4),
            ... )
            >>> tuple(fusion.fuse(per_doc).fused_state.shape)
            (2, 3, 4)
            >>> flat = FusionInput(
            ...     patient_state=torch.randn(2, 1, 3),
            ...     retrieval_memory=torch.randn(2, 1, 2, 2, 4),
            ... )
            >>> tuple(fusion.fuse(flat).fused_state.shape)
            (2, 4, 4)
            >>> fusion(
            ...     FusionInput(
            ...         patient_state=torch.zeros(2, 1, 3),
            ...         retrieval_memory=torch.zeros(2, 1, 1, 4),
            ...     )
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: ReplaceFusion expects 5D retrieval_memory...
        """
        rm = fusion_input.retrieval_memory
        if rm.ndim != 5:
            raise ValueError(f"ReplaceFusion expects 5D retrieval_memory, got shape {tuple(rm.shape)}")
        b, r, k, s_doc, d_mem = rm.shape
        if r == 1 and k == 1 and s_doc == 1:
            fused_state = rm.view(b, 1, d_mem)
        elif s_doc == 1:
            fused_state = rm.squeeze(1).squeeze(2)
        else:
            fused_state = rm.reshape(b, r * k * s_doc, d_mem)
        return FusionOutput(fused_state=fused_state)


class ConcatFusion(FusionModule):
    """Tabular fusion that concatenates patient state with retrieval memory.

    Concatenates ``patient_state`` ``(B, 1, D_ehr)`` with ``retrieval_memory``
    ``(B, 1, 1, 1, D_mem)`` (reshaped to ``(B, 1, D_mem)``) along the last
    dimension, producing ``(B, 1, D_fused)`` where ``D_fused = D_ehr + D_mem``.
    """

    def __init__(self) -> None:
        super().__init__()

    def fuse(self, fusion_input: FusionInput) -> FusionOutput:
        """Concatenate patient state and retrieval memory along the last dim.

        Args:
            fusion_input: ``FusionInput`` with:
                - ``patient_state`` shaped ``(B, 1, D_ehr)``
                - ``retrieval_memory`` shaped ``(B, 1, 1, 1, D_mem)``

        Returns:
            ``FusionOutput`` where ``fused_state`` has shape
            ``(B, 1, D_fused)``.

        Examples:
            >>> from medrap.types import FusionInput
            >>> fusion = ConcatFusion()
            >>> fusion_input = FusionInput(
            ...     patient_state=torch.FloatTensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
            ...     retrieval_memory=torch.FloatTensor([[[[[10.0, 20.0]]]], [[[[30.0, 40.0]]]]]),
            ... )
            >>> out = fusion.fuse(fusion_input)
            >>> tuple(out.fused_state.shape)
            (2, 1, 4)
            >>> out.fused_state.tolist()
            [[[1.0, 2.0, 10.0, 20.0]], [[3.0, 4.0, 30.0, 40.0]]]
            >>> tuple(fusion(fusion_input).fused_state.shape)
            (2, 1, 4)
            >>> bad = FusionInput(
            ...     patient_state=torch.randn(2, 4), retrieval_memory=torch.randn(2, 1, 1, 1, 3)
            ... )
            >>> fusion.fuse(bad)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: ConcatFusion expects patient_state shaped (B, 1, D_ehr), ...
            >>> good = FusionInput(
            ...     patient_state=torch.FloatTensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
            ...     retrieval_memory=torch.FloatTensor([[[[[10.0, 20.0]]]], [[[[30.0, 40.0]]]]]),
            ... )
            >>> tuple(fusion.fuse(good).fused_state.shape)
            (2, 1, 4)
        """
        ps = fusion_input.patient_state
        rm = fusion_input.retrieval_memory
        if ps.ndim != 3 or ps.shape[1] != 1:
            raise ValueError(
                f"ConcatFusion expects patient_state shaped (B, 1, D_ehr), got {tuple(ps.shape)}"
            )
        rm = rm.view(rm.shape[0], 1, rm.shape[-1])
        return FusionOutput(fused_state=torch.cat((ps.float(), rm.float()), dim=-1))


class _CrossAttentionLayer(nn.Module):
    """Single pre-norm cross-attention layer: LN → cross-attn → residual → LN → FFN → residual."""

    def __init__(self, d_model: int, kv_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            kdim=kv_dim,
            vdim=kv_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kvn = self.norm_kv(kv)
        attn_out, _ = self.attn(
            self.norm_q(query),
            kvn,
            kvn,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = query + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm_ff(x)))
        return x


class CrossAttentionFusion(FusionModule):
    """Transformer decoder-style fusion: patient attends to retrieved doc tokens.

    Each patient step (query) cross-attends to the full set of retrieved
    document tokens (keys/values) across ``num_layers`` stacked cross-attention
    layers, each with a residual connection, LayerNorm, and position-wise FFN.

    Args:
        d_model: Internal model dimension after input projections.
        num_heads: Number of attention heads (must divide ``d_model``).
        ff_dim: Hidden dimension of the position-wise feed-forward network.
        num_layers: Number of stacked cross-attention layers.
        d_in_patient: Input dimension of ``patient_state`` (``D_ehr``).
        d_in_doc: Input dimension of ``retrieval_memory`` token vectors (``D_mem``).
        dropout: Dropout probability applied inside attention and FFN.

    Input shapes (via :class:`~medrap.types.FusionInput`):
        - ``patient_state``:    ``(B, S_ehr, D_ehr)``
        - ``retrieval_memory``: ``(B, R, K, S_doc, D_mem)``
        - ``doc_attention_mask`` (optional): ``(B, R, K, S_doc)`` bool, ``True`` = valid token

    Output shape:
        ``fused_state``: ``(B, S_ehr, d_model)``
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        d_in_patient: int,
        d_in_doc: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patient_proj = nn.Linear(d_in_patient, d_model)
        self.layers = nn.ModuleList(
            [_CrossAttentionLayer(d_model, d_in_doc, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)

    def fuse(self, fusion_input: FusionInput) -> FusionOutput:
        """Enrich patient sequence by cross-attending to retrieved doc tokens.

        Args:
            fusion_input: ``FusionInput`` with:

                - ``patient_state`` shaped ``(B, S_ehr, D_ehr)``
                - ``retrieval_memory`` shaped ``(B, R, K, S_doc, D_mem)``
                - ``doc_attention_mask`` optional bool mask ``(B, R, K, S_doc)``

        Returns:
            ``FusionOutput`` where ``fused_state`` has shape
            ``(B, S_ehr, d_model)``.

        Raises:
            ValueError: If ``retrieval_memory`` is not 5-D.

        Examples:
            Basic forward pass — output preserves the patient sequence length:

            >>> import torch
            >>> from medrap.types import FusionInput
            >>> fusion = CrossAttentionFusion(
            ...     d_model=8,
            ...     num_heads=2,
            ...     ff_dim=16,
            ...     num_layers=2,
            ...     d_in_patient=4,
            ...     d_in_doc=6,
            ... )
            >>> fi = FusionInput(
            ...     patient_state=torch.randn(2, 3, 4),  # (B=2, S_ehr=3, D_ehr=4)
            ...     retrieval_memory=torch.randn(2, 1, 4, 5, 6),  # (B, R=1, K=4, S_doc=5, D_mem=6)
            ... )
            >>> out = fusion.fuse(fi)
            >>> tuple(out.fused_state.shape)
            (2, 3, 8)

            ``forward`` delegates to ``fuse``:

            >>> tuple(fusion(fi).fused_state.shape)
            (2, 3, 8)

            Padding tokens in retrieved docs are masked out via ``doc_attention_mask``:

            >>> mask = torch.ones(2, 1, 4, 5, dtype=torch.bool)
            >>> mask[:, :, :, -1] = False  # last token of each doc is padding
            >>> fi_masked = FusionInput(
            ...     patient_state=torch.randn(2, 3, 4),
            ...     retrieval_memory=torch.randn(2, 1, 4, 5, 6),
            ...     doc_attention_mask=mask,
            ... )
            >>> tuple(fusion.fuse(fi_masked).fused_state.shape)
            (2, 3, 8)

            Wrong ``retrieval_memory`` rank raises ``ValueError``:

            >>> bad = FusionInput(
            ...     patient_state=torch.randn(2, 3, 4),
            ...     retrieval_memory=torch.randn(2, 4, 6),
            ... )
            >>> fusion.fuse(bad)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: CrossAttentionFusion expects 5D retrieval_memory...
        """
        ps = fusion_input.patient_state
        rm = fusion_input.retrieval_memory
        if rm.ndim != 5:
            raise ValueError(f"CrossAttentionFusion expects 5D retrieval_memory, got shape {tuple(rm.shape)}")
        b, r, k, s_doc, d_mem = rm.shape

        query = self.patient_proj(ps)  # (B, S_ehr, d_model)
        kv = rm.reshape(b, r * k * s_doc, d_mem)  # (B, R*K*S_doc, D_mem)

        key_padding_mask: torch.Tensor | None = None
        if fusion_input.doc_attention_mask is not None:
            flat_mask = fusion_input.doc_attention_mask.reshape(b, r * k * s_doc).bool()
            key_padding_mask = ~flat_mask  # True = padding (PyTorch convention)

        x = query
        for layer in self.layers:
            x = layer(x, kv, key_padding_mask)

        return FusionOutput(fused_state=self.out_norm(x))
