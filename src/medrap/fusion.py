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
        """
        ps = fusion_input.patient_state
        rm = fusion_input.retrieval_memory
        if ps.ndim != 3 or ps.shape[1] != 1:
            raise ValueError(
                f"ConcatFusion expects patient_state shaped (B, 1, D_ehr), got {tuple(ps.shape)}"
            )
        rm = rm.view(rm.shape[0], 1, rm.shape[-1])
        return FusionOutput(fused_state=torch.cat((ps.float(), rm.float()), dim=-1))
