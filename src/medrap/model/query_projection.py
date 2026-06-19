"""Query projection modules for mapping patient state into retrieval space.

These components convert encoded patient representations into retrieval queries.
"""

from abc import ABC, abstractmethod

from torch import nn

from ..types import EncoderOutput, QueryOutput


class QueryProjector(nn.Module, ABC):
    """Abstract base for all query projectors.

    Subclasses must implement :meth:`project`, which maps an ``EncoderOutput``
    to a ``QueryOutput``.  The ``forward`` method delegates to ``project`` so
    that the projector can be used as a standard ``nn.Module``.
    """

    @abstractmethod
    def project(self, encoder_out: EncoderOutput) -> QueryOutput:
        """Project encoded patient state into retrieval query space.

        Args:
            encoder_out: ``EncoderOutput`` with ``patient_state`` shaped
                ``(B, S_ehr, D_ehr)`` and an optional ``attention_mask``
                shaped ``(B, S_ehr)``.

        Returns:
            A ``QueryOutput`` with ``query_embeddings`` shaped
            ``(B, R, D_ret)``.
        """

    def forward(self, encoder_out: EncoderOutput) -> QueryOutput:
        """Call ``project``."""
        return self.project(encoder_out)


class _LinearQueryProjectorBase(QueryProjector):
    """Shared ``__init__`` for query projectors backed by a single linear layer.

    Args:
        in_dim: Input patient-state size ``D_ehr``.
        out_dim: Retrieval query size ``D_ret``.
    """

    def __init__(self, *, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.linear = nn.Linear(self.in_dim, self.out_dim)


class LinearQueryProjector(_LinearQueryProjectorBase):
    """Tabular query projector producing one retrieval query per patient.

    Projects the last dimension of a ``(B, 1, D_ehr)`` patient state through a
    linear layer, producing ``(B, 1, D_ret)`` with ``R = 1``.

    Args:
        in_dim: Input patient-state size ``D_ehr``.
        out_dim: Retrieval query size ``D_ret``.
    """

    def project(self, encoder_out: EncoderOutput) -> QueryOutput:
        """Project a tabular patient state into a single retrieval query.

        Args:
            encoder_out: ``EncoderOutput`` with ``patient_state`` shaped
                ``(B, 1, D_ehr)``.

        Returns:
            ``QueryOutput`` with:
                - ``query_embeddings`` shaped ``(B, 1, D_ret)``
                - ``retrieval_step_ids=None``

        Examples:
            >>> from medrap.types import EncoderOutput
            >>> projector = LinearQueryProjector(in_dim=2, out_dim=3)
            >>> patient_state = torch.FloatTensor([[[1.0, 2.0]], [[3.0, 4.0]]])
            >>> out = projector.project(EncoderOutput(patient_state=patient_state))
            >>> tuple(out.query_embeddings.shape)
            (2, 1, 3)
            >>> out.query_embeddings.dtype
            torch.float32
            >>> out.retrieval_step_ids is None
            True
            >>> tuple(projector(EncoderOutput(patient_state=patient_state)).query_embeddings.shape)
            (2, 1, 3)
            >>> projector.project(EncoderOutput(patient_state=torch.randn(2, 4)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: LinearQueryProjector expects patient_state shaped (B, 1, D_ehr), ...
        """
        patient_state = encoder_out.patient_state
        if patient_state.ndim != 3 or patient_state.shape[1] != 1:
            raise ValueError(
                "LinearQueryProjector expects patient_state shaped (B, 1, D_ehr), "
                f"got {tuple(patient_state.shape)}"
            )
        return QueryOutput(query_embeddings=self.linear(patient_state.float()))


class SequenceMeanQueryProjector(_LinearQueryProjectorBase):
    """Sequence query projector that mean-pools over the EHR sequence.

    This is a minimal sequence baseline. It reduces ``patient_state`` across the
    sequence dimension and emits a single retrieval query per patient (``R = 1``).
    When ``encoder_out.attention_mask`` is present, padding positions are excluded
    from the mean so retrieval quality does not depend on sequence length.

    Args:
        in_dim: Input patient-state size ``D_ehr``.
        out_dim: Retrieval query size ``D_ret``.
    """

    def project(self, encoder_out: EncoderOutput) -> QueryOutput:
        """Mean-pool over sequence positions and emit one query per sample.

        Args:
            encoder_out: ``EncoderOutput`` with ``patient_state`` shaped
                ``(B, S_ehr, D_ehr)`` and an optional ``attention_mask``
                shaped ``(B, S_ehr)`` (``True`` = valid, non-padding position).

        Returns:
            ``QueryOutput`` with:
                - ``query_embeddings`` shaped ``(B, 1, D_ret)``
                - ``retrieval_step_ids=None``

        Examples:
            >>> from medrap.types import EncoderOutput
            >>> projector = SequenceMeanQueryProjector(in_dim=2, out_dim=2)
            >>> patient_state = torch.FloatTensor(
            ...     [
            ...         [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            ...         [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
            ...     ]
            ... )
            >>> out = projector.project(EncoderOutput(patient_state=patient_state))
            >>> tuple(out.query_embeddings.shape)
            (2, 1, 2)
            >>> out.query_embeddings.dtype
            torch.float32
            >>> out.retrieval_step_ids is None
            True
            >>> tuple(projector(EncoderOutput(patient_state=patient_state)).query_embeddings.shape)
            (2, 1, 2)
            >>> projector.project(EncoderOutput(patient_state=torch.randn(2, 4)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: SequenceMeanQueryProjector expects patient_state shaped ...

            Trailing padding does not change the query vector once masked:

            >>> padded_state = torch.cat([patient_state, torch.zeros(2, 1, 2)], dim=1)
            >>> mask = torch.BoolTensor([[True, True, True, False], [True, True, True, False]])
            >>> out_padded = projector.project(EncoderOutput(patient_state=padded_state, attention_mask=mask))
            >>> torch.allclose(out_padded.query_embeddings, out.query_embeddings)
            True
        """
        patient_state = encoder_out.patient_state
        if patient_state.ndim != 3:
            raise ValueError(
                "SequenceMeanQueryProjector expects patient_state shaped "
                f"(B, S_ehr, D_ehr), got {tuple(patient_state.shape)}"
            )
        attention_mask = encoder_out.attention_mask
        if attention_mask is None:
            pooled = patient_state.mean(dim=1)
        else:
            mask = attention_mask.bool().unsqueeze(-1)
            counts = mask.sum(dim=1).clamp_min(1).to(dtype=patient_state.dtype)
            pooled = (patient_state.float() * mask).sum(dim=1) / counts
        return QueryOutput(query_embeddings=self.linear(pooled).unsqueeze(1))
