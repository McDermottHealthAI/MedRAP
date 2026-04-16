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
