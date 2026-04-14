"""Prediction head modules mapping pooled states to task outputs.

These components consume pooled fused representations and produce model logits.
"""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class PredictionHead(nn.Module, ABC):
    """Abstract base for all prediction heads.

    Subclasses must implement :meth:`predict`, which maps a pooled state
    vector to task logits.  The ``forward`` method delegates to ``predict``.
    """

    @abstractmethod
    def predict(self, pooled_state: Tensor) -> Tensor:
        """Map pooled state to task logits.

        Args:
            pooled_state: Tensor with shape ``(B, D_pool)``.

        Returns:
            Tensor with shape ``(B, C)``.
        """

    def forward(self, pooled_state: Tensor) -> Tensor:
        """Call ``predict``."""
        return self.predict(pooled_state)


class LinearHead(PredictionHead):
    """Linear prediction head mapping pooled representations to task logits.

    Args:
        in_dim: Input pooled representation size ``D_pool``.
        out_dim: Output logit size ``C``.
    """

    def __init__(self, *, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.linear = nn.Linear(self.in_dim, self.out_dim)

    def predict(self, pooled_state: Tensor) -> Tensor:
        """Apply the linear prediction head.

        Args:
            pooled_state: Tensor with shape ``(B, D_pool)``.

        Returns:
            Tensor with shape ``(B, C)``.

        Examples:
            >>> head = LinearHead(in_dim=2, out_dim=3)
            >>> pooled_state = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0]])
            >>> out = head.predict(pooled_state)
            >>> tuple(out.shape)
            (2, 3)
            >>> out.dtype
            torch.float32
            >>> tuple(head(pooled_state).shape)
            (2, 3)
        """
        return self.linear(pooled_state.float())


class MLPHead(PredictionHead):
    """Two-layer MLP prediction head with ReLU activation.

    Args:
        in_dim: Input pooled representation size ``D_pool``.
        hidden_dim: Hidden layer size.
        out_dim: Output logit size ``C``.

    Examples:
        >>> head = MLPHead(in_dim=4, hidden_dim=8, out_dim=2)
        >>> pooled_state = torch.FloatTensor([[1.0, 2.0, 3.0, 4.0]])
        >>> out = head.predict(pooled_state)
        >>> tuple(out.shape)
        (1, 2)
        >>> out.dtype
        torch.float32
    """

    def __init__(self, *, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.dropout = float(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def predict(self, pooled_state: Tensor) -> Tensor:
        """Apply the MLP prediction head.

        Args:
            pooled_state: Tensor with shape ``(B, D_pool)``.

        Returns:
            Tensor with shape ``(B, C)``.
        """
        return self.mlp(pooled_state.float())
