"""Pooling modules for reducing fused representations before prediction."""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class PoolingModule(nn.Module, ABC):
    """Abstract base for all pooling modules.

    Subclasses must implement :meth:`pool`, which reduces a fused state tensor
    to a fixed-size ``(B, D_pool)`` vector.  The ``forward`` method delegates
    to ``pool``.
    """

    @abstractmethod
    def pool(self, fused_state: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Reduce fused state to a pooled vector.

        Args:
            fused_state: Tensor with shape ``(B, S_ehr, D_fused)``.
            attention_mask: Optional mask tensor.

        Returns:
            Tensor with shape ``(B, D_pool)``.
        """

    def forward(self, fused_state: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Call ``pool``."""
        return self.pool(fused_state, attention_mask=attention_mask)


class IdentityPooling(PoolingModule):
    """Tabular pooling that squeezes the singleton sequence dimension.

    Expects input shaped ``(B, 1, D_fused)`` and returns ``(B, D_pool)``
    where ``D_pool = D_fused``.
    """

    def __init__(self) -> None:
        super().__init__()

    def pool(self, fused_state: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Squeeze the singleton sequence dim from a tabular fused state.

        Args:
            fused_state: Tensor with shape ``(B, 1, D_fused)``.
            attention_mask: Ignored.

        Returns:
            Tensor with shape ``(B, D_pool)``.

        Examples:
            >>> pooling = IdentityPooling()
            >>> fused_state = torch.randn(2, 1, 4)
            >>> out = pooling.pool(fused_state)
            >>> tuple(out.shape)
            (2, 4)
            >>> tuple(pooling(fused_state).shape)
            (2, 4)
        """
        return fused_state.squeeze(1)


class MaskedMeanPooling(PoolingModule):
    """Sequence pooling module that averages over the sequence dimension."""

    def __init__(self) -> None:
        super().__init__()

    def pool(self, fused_state: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Average over the sequence dimension with an optional mask.

        Args:
            fused_state: Tensor with shape ``(B, S_ehr, D_fused)``.
            attention_mask: Optional boolean mask with shape ``(B, S_ehr)``.

        Returns:
            Tensor with shape ``(B, D_pool)``.

        Examples:
            >>> pooling = MaskedMeanPooling()
            >>> fused_state = torch.FloatTensor(
            ...     [
            ...         [[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]],
            ...         [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            ...     ]
            ... )
            >>> attention_mask = torch.BoolTensor([[True, True, False], [True, False, True]])
            >>> out = pooling.pool(fused_state, attention_mask=attention_mask)
            >>> out.tolist()
            [[2.0, 3.0], [7.0, 8.0]]
            >>> tuple(out.shape)
            (2, 2)
            >>> tuple(pooling(fused_state).shape)
            (2, 2)
            >>> pooling.pool(torch.randn(2, 4))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: MaskedMeanPooling expects fused_state shaped (B, S, D), ...
            >>> pooling.pool(fused_state, attention_mask=torch.ones(3, 3, dtype=torch.bool))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: attention_mask must match the first two fused_state dimensions. ...
        """
        if fused_state.ndim != 3:
            raise ValueError(
                f"MaskedMeanPooling expects fused_state shaped (B, S, D), got {tuple(fused_state.shape)}"
            )
        if attention_mask is None:
            return fused_state.float().mean(dim=1)

        mask = attention_mask.bool()
        if mask.shape != fused_state.shape[:2]:
            raise ValueError(
                "attention_mask must match the first two fused_state dimensions. "
                f"Got mask={tuple(mask.shape)}, fused_state={tuple(fused_state.shape)}"
            )
        expanded_mask = mask.unsqueeze(-1)
        counts = expanded_mask.sum(dim=1).clamp_min(1).to(dtype=fused_state.dtype)
        return (fused_state.float() * expanded_mask).sum(dim=1) / counts
