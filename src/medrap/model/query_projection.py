"""Query projection modules for mapping patient state into retrieval space.

These components convert encoded patient representations into retrieval queries.
"""

from abc import ABC, abstractmethod

from torch import Tensor, nn

from ..types import QueryOutput


class QueryProjector(nn.Module, ABC):
    """Abstract base for all query projectors.

    Subclasses must implement :meth:`project`, which maps an encoded patient
    state to a ``QueryOutput``.  The ``forward`` method delegates to
    ``project`` so that the projector can be used as a standard ``nn.Module``.
    """

    @abstractmethod
    def project(self, patient_state: Tensor) -> QueryOutput:
        """Project patient state into retrieval query space.

        Args:
            patient_state: Encoded patient tensor with shape
                ``(B, S_ehr, D_ehr)``.

        Returns:
            A ``QueryOutput`` with ``query_embeddings`` shaped
            ``(B, R, D_ret)``.
        """

    def forward(self, patient_state: Tensor) -> QueryOutput:
        """Call ``project``."""
        return self.project(patient_state)


class LinearQueryProjector(QueryProjector):
    """Tabular query projector producing one retrieval query per patient.

    Projects the last dimension of a ``(B, 1, D_ehr)`` patient state through a
    linear layer, producing ``(B, 1, D_ret)`` with ``R = 1``.

    Args:
        in_dim: Input patient-state size ``D_ehr``.
        out_dim: Retrieval query size ``D_ret``.
    """

    def __init__(self, *, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.linear = nn.Linear(self.in_dim, self.out_dim)

    def project(self, patient_state: Tensor) -> QueryOutput:
        """Project a tabular patient state into a single retrieval query.

        Args:
            patient_state: Tensor with shape ``(B, 1, D_ehr)``.

        Returns:
            ``QueryOutput`` with:
                - ``query_embeddings`` shaped ``(B, 1, D_ret)``
                - ``retrieval_step_ids=None``

        Examples:
            >>> projector = LinearQueryProjector(in_dim=2, out_dim=3)
            >>> patient_state = torch.FloatTensor([[[1.0, 2.0]], [[3.0, 4.0]]])
            >>> out = projector.project(patient_state)
            >>> tuple(out.query_embeddings.shape)
            (2, 1, 3)
            >>> out.query_embeddings.dtype
            torch.float32
            >>> out.retrieval_step_ids is None
            True
            >>> tuple(projector(patient_state).query_embeddings.shape)
            (2, 1, 3)
            >>> projector.project(torch.randn(2, 4))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: LinearQueryProjector expects patient_state shaped (B, 1, D_ehr), ...
        """
        if patient_state.ndim != 3 or patient_state.shape[1] != 1:
            raise ValueError(
                "LinearQueryProjector expects patient_state shaped (B, 1, D_ehr), "
                f"got {tuple(patient_state.shape)}"
            )
        return QueryOutput(query_embeddings=self.linear(patient_state.float()))


class SequenceMeanQueryProjector(QueryProjector):
    """Sequence query projector that mean-pools over the EHR sequence.

    This is a minimal sequence baseline. It reduces ``patient_state`` across the
    sequence dimension and emits a single retrieval query per patient (``R = 1``).

    Args:
        in_dim: Input patient-state size ``D_ehr``.
        out_dim: Retrieval query size ``D_ret``.
    """

    def __init__(self, *, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.linear = nn.Linear(self.in_dim, self.out_dim)

    def project(self, patient_state: Tensor) -> QueryOutput:
        """Mean-pool over sequence positions and emit one query per sample.

        Args:
            patient_state: Encoded patient tensor with shape
                ``(B, S_ehr, D_ehr)``.

        Returns:
            ``QueryOutput`` with:
                - ``query_embeddings`` shaped ``(B, 1, D_ret)``
                - ``retrieval_step_ids=None``

        Examples:
            >>> projector = SequenceMeanQueryProjector(in_dim=2, out_dim=2)
            >>> patient_state = torch.FloatTensor(
            ...     [
            ...         [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            ...         [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
            ...     ]
            ... )
            >>> out = projector.project(patient_state)
            >>> tuple(out.query_embeddings.shape)
            (2, 1, 2)
            >>> out.query_embeddings.dtype
            torch.float32
            >>> out.retrieval_step_ids is None
            True
            >>> tuple(projector(patient_state).query_embeddings.shape)
            (2, 1, 2)
            >>> projector.project(torch.randn(2, 4))
            Traceback (most recent call last):
                ...
            ValueError: SequenceMeanQueryProjector expects patient_state shaped ...
        """
        if patient_state.ndim != 3:
            raise ValueError(
                "SequenceMeanQueryProjector expects patient_state shaped "
                f"(B, S_ehr, D_ehr), got {tuple(patient_state.shape)}"
            )
        pooled = patient_state.mean(dim=1)
        return QueryOutput(query_embeddings=self.linear(pooled).unsqueeze(1))
