"""Fusion modules for combining patient and retrieval representations.

These components define how retrieved memory contributes to downstream model states before pooling and
prediction.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn

from ..types import FusionInput, FusionOutput


class FusionModule(nn.Module, ABC):
    """Abstract base for all fusion modules.

    Subclasses must implement :meth:`fuse`, which combines patient state and
    retrieval memory into a fused representation.  The ``forward`` method
    delegates to ``fuse`` so that the module can be used as a standard
    ``nn.Module``.

    Attributes:
        produces_per_document_state: Whether ``fused_state`` is indexed by
            retrieved document (``(B, K, D)``) rather than by EHR sequence
            position (``(B, S_ehr, D)``). :class:`~medrap.model.model.RetrievalAugmentedModel`
            requires this to be ``True`` when ``marginalized_retrieval=True``,
            since the marginalization sums over dim 1 expecting it to range
            over retrieved documents.
    """

    produces_per_document_state: bool = False

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
    - Per-document keys ``(B, 1, K, 1, D_mem)`` -> ``fused_state`` ``(B, K, D_mem)``
      when ``R = 1`` and the singleton doc-length dim is 1; otherwise (``R > 1``
      and/or ``S_doc > 1``) ``(B, R * K * S_doc, D_mem)``.
    """

    produces_per_document_state = True

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
            >>> multi_round = FusionInput(
            ...     patient_state=torch.randn(2, 1, 3),
            ...     retrieval_memory=torch.randn(2, 2, 1, 1, 4),
            ... )
            >>> tuple(fusion.fuse(multi_round).fused_state.shape)
            (2, 2, 4)
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
        elif r == 1 and s_doc == 1:
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


class PassthroughFusion(FusionModule):
    """No-op fusion that ignores retrieval memory and returns patient state unchanged.

    Use this as a retrieval-ablation baseline: the retriever still runs, but its
    output is discarded before the prediction head, so the model predicts from
    patient state alone.

    Input shapes (via :class:`~medrap.types.FusionInput`):
        - ``patient_state``: ``(B, S_ehr, D_ehr)``

    Output shape:
        ``fused_state``: ``(B, S_ehr, D_ehr)`` — identical to ``patient_state``.
    """

    def __init__(self) -> None:
        super().__init__()

    def fuse(self, fusion_input: FusionInput) -> FusionOutput:
        """Return patient state as-is, ignoring retrieval memory.

        Examples:
            >>> from medrap.types import FusionInput
            >>> fusion = PassthroughFusion()
            >>> ps = torch.randn(2, 5, 128)
            >>> fi = FusionInput(
            ...     patient_state=ps,
            ...     retrieval_memory=torch.randn(2, 1, 8, 10, 64),
            ... )
            >>> out = fusion.fuse(fi)
            >>> tuple(out.fused_state.shape)
            (2, 5, 128)
            >>> (out.fused_state == ps).all().item()
            True

            ``patient_attention_mask`` is forwarded unchanged, since the fused
            sequence dim is still indexed by (possibly padded) EHR position:

            >>> mask = torch.BoolTensor([[True, True, True, False, False]] * 2)
            >>> fi_masked = FusionInput(
            ...     patient_state=ps,
            ...     retrieval_memory=torch.randn(2, 1, 8, 10, 64),
            ...     patient_attention_mask=mask,
            ... )
            >>> (fusion.fuse(fi_masked).attention_mask == mask).all().item()
            True
        """
        return FusionOutput(
            fused_state=fusion_input.patient_state,
            attention_mask=fusion_input.patient_attention_mask,
        )


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

            ``patient_attention_mask`` is forwarded unchanged, since the output
            sequence dim is still indexed by (possibly padded) EHR position:

            >>> patient_mask = torch.BoolTensor([[True, True, False]] * 2)
            >>> fi_patient_masked = FusionInput(
            ...     patient_state=torch.randn(2, 3, 4),
            ...     retrieval_memory=torch.randn(2, 1, 4, 5, 6),
            ...     patient_attention_mask=patient_mask,
            ... )
            >>> (fusion.fuse(fi_patient_masked).attention_mask == patient_mask).all().item()
            True

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

        return FusionOutput(fused_state=self.out_norm(x), attention_mask=fusion_input.patient_attention_mask)


class PerDocCrossAttentionFusion(FusionModule):
    """Cross-attention fusion that attends to each retrieved document independently.

    Unlike :class:`CrossAttentionFusion` — which attends jointly to all ``K``
    documents and produces ``(B, S_ehr, d_model)`` indexed by EHR sequence
    position — this module expands the batch to ``B * K``, runs cross-attention
    between each patient and exactly one of its retrieved documents, then
    mean-pools over the patient sequence to produce one vector per document.

    The resulting ``(B, K, d_model)`` output is indexed by retrieved document,
    so it is compatible with REALM-style marginalized retrieval
    (:attr:`~medrap.model.model.RetrievalAugmentedModel.marginalized_retrieval`)
    at any value of ``K`` — unlike ``CrossAttentionFusion``, which marginalizes
    over EHR sequence positions instead of documents when paired with that flag.

    Only ``R = 1`` retrieval step is supported.

    Args:
        d_model: Internal model dimension after input projection.
        num_heads: Number of attention heads (must divide ``d_model``).
        ff_dim: Hidden dimension of the position-wise feed-forward network.
        num_layers: Number of stacked cross-attention layers.
        d_in_patient: Input dimension of ``patient_state`` (``D_ehr``).
        d_in_doc: Input dimension of ``retrieval_memory`` token vectors (``D_mem``).
        dropout: Dropout probability applied inside attention and FFN.
        zero_init_attn_out: Start every cross-attention output projection at zero, so at
            initialization documents add nothing to the patient state (it passes through the
            residual) and their contribution grows only as training finds it useful. Unlike a
            gate there is no single switch for turning documents off.

    Input shapes (via :class:`~medrap.types.FusionInput`):
        - ``patient_state``:    ``(B, S_ehr, D_ehr)``
        - ``retrieval_memory``: ``(B, 1, K, S_doc, D_mem)``
        - ``doc_attention_mask`` (optional): ``(B, 1, K, S_doc)`` bool, ``True`` = valid token

    Output shape:
        ``fused_state``: ``(B, K, d_model)``
    """

    produces_per_document_state = True

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
        zero_init_attn_out: bool = False,
    ) -> None:
        super().__init__()
        self.patient_proj = nn.Linear(d_in_patient, d_model)
        self.layers = nn.ModuleList(
            [_CrossAttentionLayer(d_model, d_in_doc, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        if zero_init_attn_out:
            for layer in self.layers:
                nn.init.zeros_(layer.attn.out_proj.weight)
                nn.init.zeros_(layer.attn.out_proj.bias)

    def fuse(self, fusion_input: FusionInput) -> FusionOutput:
        """Enrich patient state by attending to each retrieved doc independently.

        Args:
            fusion_input: ``FusionInput`` with:

                - ``patient_state`` shaped ``(B, S_ehr, D_ehr)``
                - ``retrieval_memory`` shaped ``(B, 1, K, S_doc, D_mem)``
                - ``doc_attention_mask`` optional bool mask ``(B, 1, K, S_doc)``

        Returns:
            ``FusionOutput`` where ``fused_state`` has shape ``(B, K, d_model)``.

        Raises:
            ValueError: If ``retrieval_memory`` is not 5-D or ``R != 1``.

        Examples:
            Basic forward pass — output is indexed by document, not sequence position:

            >>> import torch
            >>> from medrap.types import FusionInput
            >>> fusion = PerDocCrossAttentionFusion(
            ...     d_model=8,
            ...     num_heads=2,
            ...     ff_dim=16,
            ...     num_layers=2,
            ...     d_in_patient=4,
            ...     d_in_doc=6,
            ... )
            >>> fi = FusionInput(
            ...     patient_state=torch.randn(2, 10, 4),  # (B=2, S_ehr=10, D_ehr=4)
            ...     retrieval_memory=torch.randn(2, 1, 3, 5, 6),  # (B, R=1, K=3, S_doc=5, D_mem=6)
            ... )
            >>> out = fusion.fuse(fi)
            >>> tuple(out.fused_state.shape)
            (2, 3, 8)

            ``forward`` delegates to ``fuse``:

            >>> tuple(fusion(fi).fused_state.shape)
            (2, 3, 8)

            Doc padding tokens are masked out of attention:

            >>> mask = torch.ones(2, 1, 3, 5, dtype=torch.bool)
            >>> mask[:, :, :, -1] = False  # last token of each doc is padding
            >>> fi_masked = FusionInput(
            ...     patient_state=torch.randn(2, 10, 4),
            ...     retrieval_memory=torch.randn(2, 1, 3, 5, 6),
            ...     doc_attention_mask=mask,
            ... )
            >>> tuple(fusion.fuse(fi_masked).fused_state.shape)
            (2, 3, 8)

            With ``zero_init_attn_out`` the documents add nothing at initialization, so
            every document gives the same vector:

            >>> zero = PerDocCrossAttentionFusion(
            ...     d_model=8,
            ...     num_heads=2,
            ...     ff_dim=16,
            ...     num_layers=2,
            ...     d_in_patient=4,
            ...     d_in_doc=6,
            ...     zero_init_attn_out=True,
            ... )
            >>> same = zero.fuse(fi).fused_state
            >>> torch.allclose(same[:, 0], same[:, 1], atol=1e-6) and torch.allclose(
            ...     same[:, 0], same[:, 2], atol=1e-6
            ... )
            True

            Patient padding is left out of the mean over codes, so padding a
            patient's sequence does not change its per-document vectors:

            >>> _ = fusion.eval()
            >>> ps, rm = torch.randn(1, 4, 4), torch.randn(1, 1, 3, 5, 6)
            >>> short = fusion.fuse(FusionInput(patient_state=ps, retrieval_memory=rm))
            >>> padded = fusion.fuse(
            ...     FusionInput(
            ...         patient_state=torch.cat([ps, torch.zeros(1, 6, 4)], dim=1),
            ...         retrieval_memory=rm,
            ...         patient_attention_mask=torch.BoolTensor([[True] * 4 + [False] * 6]),
            ...     )
            ... )
            >>> torch.allclose(short.fused_state, padded.fused_state, atol=1e-6)
            True

            Wrong ``retrieval_memory`` rank raises ``ValueError``:

            >>> bad_rank = FusionInput(
            ...     patient_state=torch.randn(2, 10, 4),
            ...     retrieval_memory=torch.randn(2, 3, 6),
            ... )
            >>> fusion.fuse(bad_rank)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: PerDocCrossAttentionFusion expects 5D retrieval_memory...

            ``R != 1`` raises ``ValueError``:

            >>> bad = FusionInput(
            ...     patient_state=torch.randn(2, 10, 4),
            ...     retrieval_memory=torch.randn(2, 2, 3, 5, 6),
            ... )
            >>> fusion.fuse(bad)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: PerDocCrossAttentionFusion only supports R=1...
        """
        ps = fusion_input.patient_state  # (B, S_ehr, D_ehr)
        rm = fusion_input.retrieval_memory  # (B, R, K, S_doc, D_mem)

        if rm.ndim != 5:
            raise ValueError(
                f"PerDocCrossAttentionFusion expects 5D retrieval_memory, got shape {tuple(rm.shape)}"
            )
        b, r, k, s_doc, d_mem = rm.shape
        if r != 1:
            raise ValueError(f"PerDocCrossAttentionFusion only supports R=1, got R={r}")

        # Pair each patient with each of its K docs: [p0, p0, ..., p1, p1, ...] along dim 0.
        ps_expanded = ps.repeat_interleave(k, dim=0)  # (B*K, S_ehr, D_ehr)
        kv = rm.squeeze(1).reshape(b * k, s_doc, d_mem)  # (B*K, S_doc, D_mem)

        key_padding_mask: torch.Tensor | None = None
        if fusion_input.doc_attention_mask is not None:
            flat_mask = fusion_input.doc_attention_mask.squeeze(1).reshape(b * k, s_doc).bool()
            key_padding_mask = ~flat_mask  # True = padding (PyTorch convention)

        x = self.patient_proj(ps_expanded)  # (B*K, S_ehr, d_model)
        for layer in self.layers:
            x = layer(x, kv, key_padding_mask)
        x = self.out_norm(x)  # (B*K, S_ehr, d_model)

        if fusion_input.patient_attention_mask is None:
            pooled = x.mean(dim=1)  # (B*K, d_model) — one vector per (patient, doc) pair
        else:
            # Mean over valid patient codes only, matching MaskedMeanPooling on the patient-only path.
            valid = fusion_input.patient_attention_mask.repeat_interleave(k, dim=0).unsqueeze(-1).to(x.dtype)
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        fused_state = pooled.reshape(b, k, -1)  # (B, K, d_model)

        return FusionOutput(fused_state=fused_state)


class GatedDocFusion(FusionModule):
    """Per-document fusion that starts as the patient-only model and adds documents through a gate.

    For each retrieved document ``k`` one query vector cross-attends to the document's token
    vectors, then a feed-forward layer gives the document's contribution ``c_k``. The query
    vector is a linear map of ``h``, the masked mean of the patient states (``mode="patient"``),
    or of the retrieval query (``mode="query"``, retrieval only). The output per document is

    - ``mode="patient"``: ``h + tanh(g) * c_k``, with the scalar gate ``g`` starting at
      ``gate_init`` (0 by default, as in Flamingo's gated cross-attention). At ``g = 0`` every
      document gives exactly ``h``, the input the patient-only model's head reads, so training
      starts at the patient-only model and uses a document only if it lowers the loss.
    - ``mode="query"``: ``c_k``. The patient reaches the output only through the retrieval
      query, so this measures what the documents carry by themselves.

    ``c_k`` has no residual from the query vector, so it holds only document content. The output
    width is ``d_in_patient`` in both modes. Pair with ``marginalized_retrieval=True``.

    Args:
        d_in_patient: Width ``W`` of ``patient_state``; also the output width.
        d_in_doc: Width of the document token vectors in ``retrieval_memory``.
        num_heads: Attention heads (must divide ``d_in_patient``).
        mode: ``"patient"`` or ``"query"``.
        d_query: Width of ``query_embeddings`` (used in ``mode="query"``).
        gate_init: Starting value of ``g`` (``mode="patient"``).
        dropout: Dropout in attention and the feed-forward layer.

    Input shapes (via :class:`~medrap.types.FusionInput`):
        - ``patient_state``: ``(B, S_ehr, W)``
        - ``retrieval_memory``: ``(B, 1, K, S_doc, d_in_doc)``
        - ``doc_attention_mask`` (optional): ``(B, 1, K, S_doc)``, ``True`` = valid token
        - ``patient_attention_mask`` (optional): ``(B, S_ehr)``, ``True`` = valid code
        - ``query_embeddings`` (``mode="query"``): ``(B, 1, d_query)``

    Output shape:
        ``fused_state``: ``(B, K, W)``

    Examples:
        >>> import torch
        >>> from medrap.types import FusionInput
        >>> _ = torch.manual_seed(0)
        >>> fusion = GatedDocFusion(d_in_patient=4, d_in_doc=6, num_heads=2)
        >>> ps = torch.randn(2, 5, 4)
        >>> mask = torch.BoolTensor([[True] * 5, [True, True, False, False, False]])
        >>> fi = FusionInput(
        ...     patient_state=ps,
        ...     retrieval_memory=torch.randn(2, 1, 3, 7, 6),
        ...     patient_attention_mask=mask,
        ... )
        >>> out = fusion(fi).fused_state
        >>> tuple(out.shape)
        (2, 3, 4)

        With the gate at 0 every document gives the masked mean of the patient states:

        >>> h = (ps * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        >>> torch.allclose(out, h.unsqueeze(1).expand(2, 3, 4))
        True

        Once the gate opens, documents change the output, and the gate gets a gradient at 0:

        >>> fusion.gate.grad is None
        True
        >>> fusion(fi).fused_state.sum().backward()
        >>> bool(fusion.gate.grad.abs() > 0)
        True
        >>> with torch.no_grad():
        ...     _ = fusion.gate.fill_(1.0)
        >>> torch.allclose(fusion(fi).fused_state, h.unsqueeze(1).expand(2, 3, 4))
        False

        Document padding is masked out of attention:

        >>> doc_mask = torch.ones(2, 1, 3, 7, dtype=torch.bool)
        >>> doc_mask[..., 3:] = False
        >>> rm = torch.randn(2, 1, 3, 7, 6)
        >>> a = fusion(FusionInput(patient_state=ps, retrieval_memory=rm, doc_attention_mask=doc_mask))
        >>> rm2 = rm.clone()
        >>> rm2[..., 3:, :] = 100.0
        >>> b = fusion(FusionInput(patient_state=ps, retrieval_memory=rm2, doc_attention_mask=doc_mask))
        >>> torch.allclose(a.fused_state, b.fused_state)
        True

        Retrieval only (``mode="query"``) reads the documents with the retrieval query and does
        not depend on the patient states:

        >>> only = GatedDocFusion(d_in_patient=4, d_in_doc=6, num_heads=2, mode="query", d_query=8)
        >>> q = torch.randn(2, 1, 8)
        >>> rm = torch.randn(2, 1, 3, 7, 6)
        >>> x = only(FusionInput(patient_state=ps, retrieval_memory=rm, query_embeddings=q))
        >>> y = only(FusionInput(patient_state=ps * 0, retrieval_memory=rm, query_embeddings=q))
        >>> tuple(x.fused_state.shape), torch.allclose(x.fused_state, y.fused_state)
        ((2, 3, 4), True)

        Errors:

        >>> only(fi)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: GatedDocFusion(mode='query') needs query_embeddings...
        >>> GatedDocFusion(d_in_patient=4, d_in_doc=6, num_heads=2, mode="other")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: mode must be 'patient' or 'query'...
        >>> fusion(FusionInput(patient_state=ps, retrieval_memory=torch.randn(2, 3, 6)))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: GatedDocFusion expects retrieval_memory (B, 1, K, S_doc, D)...
    """

    produces_per_document_state = True

    def __init__(
        self,
        *,
        d_in_patient: int,
        d_in_doc: int,
        num_heads: int,
        mode: str = "patient",
        d_query: int = 1024,
        gate_init: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in ("patient", "query"):
            raise ValueError(f"mode must be 'patient' or 'query', got {mode!r}")
        self.mode = mode
        self.query_proj = nn.Linear(d_in_patient if mode == "patient" else d_query, d_in_patient)
        self.norm_kv = nn.LayerNorm(d_in_doc)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_in_patient,
            num_heads=num_heads,
            kdim=d_in_doc,
            vdim=d_in_doc,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_in_patient)
        self.ff = nn.Sequential(
            nn.Linear(d_in_patient, 2 * d_in_patient),
            nn.GELU(),
            nn.Linear(2 * d_in_patient, d_in_patient),
        )
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def fuse(self, fusion_input: FusionInput) -> FusionOutput:
        """Return one ``W``-wide vector per (patient, retrieved document); see the class docstring."""
        ps = fusion_input.patient_state
        rm = fusion_input.retrieval_memory
        if rm.ndim != 5 or rm.shape[1] != 1:
            raise ValueError(
                f"GatedDocFusion expects retrieval_memory (B, 1, K, S_doc, D), got {tuple(rm.shape)}"
            )
        b, _, k, s_doc, d_mem = rm.shape

        if fusion_input.patient_attention_mask is None:
            h = ps.mean(dim=1)
        else:
            valid = fusion_input.patient_attention_mask.unsqueeze(-1).to(ps.dtype)
            h = (ps * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)  # (B, W)

        if self.mode == "patient":
            source = h
        else:
            if fusion_input.query_embeddings is None:
                raise ValueError("GatedDocFusion(mode='query') needs query_embeddings in FusionInput")
            source = fusion_input.query_embeddings[:, 0]  # (B, d_query)
        query = self.query_proj(source).repeat_interleave(k, dim=0).unsqueeze(1)  # (B*K, 1, W)

        kv = self.norm_kv(rm.reshape(b * k, s_doc, d_mem))
        key_padding_mask = None
        if fusion_input.doc_attention_mask is not None:
            key_padding_mask = ~fusion_input.doc_attention_mask.reshape(b * k, s_doc).bool()
        c, _ = self.attn(query, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        c = c + self.drop(self.ff(self.norm_ff(c)))
        c = self.drop(c).reshape(b, k, -1)  # (B, K, W)

        if self.mode == "query":
            return FusionOutput(fused_state=c)
        return FusionOutput(fused_state=h.unsqueeze(1) + torch.tanh(self.gate) * c)
