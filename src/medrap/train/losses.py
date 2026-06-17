"""This module provides loss functions for retrieval-augmented pretraining."""

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .task import SupervisedLoss, TaskTargets
from ..types import ModelOutput


class MultiTaskBCELoss(SupervisedLoss):
    """Masked mean BCE-with-logits loss across N simultaneous binary tasks.

    Expects logits of shape ``(B, N)`` and float targets of shape ``(B, N)``
    where ``NaN`` entries are excluded from the loss.

    Examples:
        >>> import torch
        >>> from medrap.types import ModelOutput
        >>> loss_fn = MultiTaskBCELoss()
        >>> preds = ModelOutput(logits=torch.zeros(2, 3))
        >>> targets = torch.tensor([[1.0, 0.0, float("nan")], [0.0, 1.0, 1.0]])
        >>> loss = loss_fn(preds, targets)
        >>> loss.shape
        torch.Size([])
        >>> float(loss) > 0
        True

        All-NaN batch raises no error but returns zero loss:

        >>> all_nan = torch.full((2, 3), float("nan"))
        >>> loss_fn(ModelOutput(logits=torch.zeros(2, 3)), all_nan).item()
        0.0
    """

    def forward(self, predictions: ModelOutput, targets: TaskTargets) -> Tensor:
        """Compute masked mean BCE across all valid (patient, task) pairs.

        Args:
            predictions: Model output with logits shaped ``(B, N)``.
            targets: Float tensor shaped ``(B, N)``, NaN entries are masked out.

        Returns:
            Scalar loss.
        """
        if not isinstance(targets, Tensor):
            raise ValueError("MultiTaskBCELoss expects tensor targets.")
        logits = predictions.logits  # (B, N)
        valid = ~targets.isnan()  # (B, N)
        labels = targets.nan_to_num(0.0)
        per_element = functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        denom = valid.float().sum()
        if denom == 0:
            return per_element.sum() * 0.0
        return (per_element * valid.float()).sum() / denom


class MultiTaskBCEMarginalizedLoss(SupervisedLoss):
    """REALM-style marginalized binary BCE for N simultaneous binary tasks.

    For each (patient, task) pair the binary prediction probability is
    marginalized over K retrieved documents weighted by retriever scores::

        P(y_n = 1 | x) = sum_k  P_ret(doc_k | x) * sigmoid(logit_{k,n})

    where ``P_ret = softmax(doc_scores)``.  The per-pair loss is then the
    standard binary cross-entropy against this marginal probability, averaged
    over all valid pairs (``NaN`` targets are excluded).

    Because document scores receive gradients through the marginalization,
    the retriever is trained end-to-end.

    Requires ``marginalized_retrieval=True`` on :class:`~medrap.model.RetrievalAugmentedModel`
    so that ``predictions.metadata`` contains:

    * ``per_doc_logits``: ``(B, K, N)``
    * ``differentiable_doc_scores``: ``(B, K)``

    Args:
        num_tasks: Number of binary prediction tasks N.

    Examples:
        >>> import torch
        >>> from medrap.types import ModelOutput
        >>> loss_fn = MultiTaskBCEMarginalizedLoss(num_tasks=2)
        >>> B, K, N = 2, 4, 2
        >>> per_doc = torch.randn(B, K, N)
        >>> scores = torch.randn(B, K)
        >>> targets = torch.tensor([[1.0, 0.0], [float("nan"), 1.0]])
        >>> pred = ModelOutput(
        ...     logits=torch.zeros(B, N),
        ...     metadata={"per_doc_logits": per_doc, "differentiable_doc_scores": scores},
        ... )
        >>> loss = loss_fn(pred, targets)
        >>> loss.shape
        torch.Size([])
        >>> float(loss) > 0
        True

        With K=1 the marginalization reduces to standard masked BCE:

        >>> loss_fn1 = MultiTaskBCEMarginalizedLoss(num_tasks=2)
        >>> per_doc1 = torch.zeros(2, 1, 2)
        >>> pred1 = ModelOutput(
        ...     logits=torch.zeros(2, 2),
        ...     metadata={"per_doc_logits": per_doc1, "differentiable_doc_scores": torch.zeros(2, 1)},
        ... )
        >>> targets1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        >>> marginal = loss_fn1(pred1, targets1)
        >>> plain = MultiTaskBCELoss()(pred1, targets1)
        >>> torch.isclose(marginal, plain, atol=1e-5)
        tensor(True)

        All-NaN batch returns zero:

        >>> loss_fn2 = MultiTaskBCEMarginalizedLoss(num_tasks=2)
        >>> miss = torch.full((2, 2), float("nan"))
        >>> pred2 = ModelOutput(
        ...     logits=torch.zeros(2, 2),
        ...     metadata={
        ...         "per_doc_logits": torch.randn(2, 4, 2),
        ...         "differentiable_doc_scores": torch.randn(2, 4),
        ...     },
        ... )
        >>> abs(loss_fn2(pred2, miss).item())
        0.0
    """

    def __init__(self, *, num_tasks: int) -> None:
        super().__init__()
        self.num_tasks = num_tasks

    def forward(self, predictions: ModelOutput, targets: TaskTargets) -> Tensor:
        """Compute REALM-style marginalized BCE across all valid (patient, task) pairs.

        Args:
            predictions: ``ModelOutput`` with metadata containing
                ``per_doc_logits (B, K, N)`` and
                ``differentiable_doc_scores (B, K)``.
            targets: Float tensor ``(B, N)``, NaN entries masked out.

        Returns:
            Scalar loss.
        """
        if not isinstance(targets, Tensor):
            raise ValueError("MultiTaskBCEMarginalizedLoss expects tensor targets.")

        meta = predictions.metadata
        per_doc_logits = meta.get("per_doc_logits")
        doc_scores = meta.get("differentiable_doc_scores")
        if not isinstance(per_doc_logits, Tensor):
            raise ValueError(
                "MultiTaskBCEMarginalizedLoss requires marginalized_retrieval=True; "
                "metadata missing 'per_doc_logits'."
            )
        if not isinstance(doc_scores, Tensor):
            raise ValueError(
                "MultiTaskBCEMarginalizedLoss requires marginalized_retrieval=True; "
                "metadata missing 'differentiable_doc_scores'."
            )

        # log P_ret: (B, K, 1) — will broadcast over N tasks
        log_p_ret = functional.log_softmax(doc_scores, dim=-1).unsqueeze(-1)

        # log marginal P(y=1|x): logsumexp_k(log P_ret_k + log_sigmoid(logit_k))
        log_p_pos = torch.logsumexp(log_p_ret + functional.logsigmoid(per_doc_logits), dim=1)
        # log marginal P(y=0|x): logsumexp_k(log P_ret_k + log_sigmoid(-logit_k))
        log_p_neg = torch.logsumexp(log_p_ret + functional.logsigmoid(-per_doc_logits), dim=1)

        valid = ~targets.isnan()
        safe_targets = targets.nan_to_num(0.0)

        per_element = -(safe_targets * log_p_pos + (1.0 - safe_targets) * log_p_neg)
        denom = valid.float().sum()
        if denom == 0:
            return per_element.sum() * 0.0
        return (per_element * valid.float()).sum() / denom


class MarginalizedRetrievalLoss(nn.Module):
    """REALM-style marginalized log-likelihood loss over retrieved documents.

    Marginalizes the prediction over retrieved documents weighted by retriever
    probabilities::

        L = -log(Σ_{k=1}^{K} P_ret(doc_k | x) * P_pred(y | x, doc_k))

    where:

        P_ret(doc_k | x)      = softmax(doc_scores)_k
        P_pred(y | x, doc_k)  = softmax(per_doc_logits_k)[y]

    Examples:
        >>> loss_fn = MarginalizedRetrievalLoss()
        >>> per_doc_logits = torch.randn(2, 3, 4)
        >>> doc_scores = torch.randn(2, 3)
        >>> targets = torch.randint(0, 4, (2,))
        >>> loss = loss_fn(per_doc_logits=per_doc_logits, doc_scores=doc_scores, targets=targets)
        >>> loss.shape
        torch.Size([])

        With K=1 the marginalization reduces to standard cross-entropy:

        >>> per_doc_logits = torch.randn(4, 1, 5)
        >>> targets = torch.randint(0, 5, (4,))
        >>> loss = MarginalizedRetrievalLoss()(
        ...     per_doc_logits=per_doc_logits,
        ...     doc_scores=torch.zeros(4, 1),
        ...     targets=targets,
        ... )
        >>> expected = torch.nn.functional.cross_entropy(per_doc_logits.squeeze(1), targets)
        >>> torch.isclose(loss, expected, atol=1e-5)
        tensor(True)

        When the retriever concentrates on one document, the loss approaches
        that document's cross-entropy:

        >>> per_doc_logits = torch.tensor([[[2.0, -2.0], [-2.0, 2.0]]])
        >>> doc_scores = torch.tensor([[100.0, -100.0]])
        >>> targets = torch.tensor([0])
        >>> loss = MarginalizedRetrievalLoss()(
        ...     per_doc_logits=per_doc_logits,
        ...     doc_scores=doc_scores,
        ...     targets=targets,
        ... )
        >>> expected = functional.cross_entropy(per_doc_logits[:, 0, :], targets)
        >>> torch.isclose(loss, expected, atol=1e-3)
        tensor(True)

        Gradients flow through both ``per_doc_logits`` and ``doc_scores``:

        >>> per_doc_logits = torch.randn(2, 3, 4, requires_grad=True)
        >>> doc_scores = torch.randn(2, 3, requires_grad=True)
        >>> loss = MarginalizedRetrievalLoss()(
        ...     per_doc_logits=per_doc_logits,
        ...     doc_scores=doc_scores,
        ...     targets=torch.tensor([0, 2]),
        ... )
        >>> loss.backward()
        >>> per_doc_logits.grad is not None and doc_scores.grad is not None
        True
    """

    def forward(
        self,
        *,
        per_doc_logits: Tensor,
        doc_scores: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Compute the marginalized retrieval loss.

        Args:
            per_doc_logits: Per-document prediction logits with shape
                ``(B, K, C)`` where K is the number of retrieved documents
                and C is the number of classes.
            doc_scores: Retrieval scores with shape ``(B, K)``.
            targets: Ground-truth class indices with shape ``(B,)``.

        Returns:
            Scalar mean loss over the batch.

        Examples:
            >>> loss_fn = MarginalizedRetrievalLoss()
            >>> per_doc_logits = torch.tensor([[[-1.0, 1.0], [1.0, -1.0]]])
            >>> doc_scores = torch.tensor([[0.0, 0.0]])
            >>> targets = torch.tensor([0])
            >>> loss = loss_fn(per_doc_logits=per_doc_logits, doc_scores=doc_scores, targets=targets)
            >>> torch.isclose(loss, torch.log(torch.tensor(2.0)), atol=1e-4)
            tensor(True)
        """
        batch_size, num_docs, _ = per_doc_logits.shape

        log_p_ret = functional.log_softmax(doc_scores, dim=-1)
        log_p_pred = functional.log_softmax(per_doc_logits, dim=-1)

        target_idx = targets.unsqueeze(1).unsqueeze(2).expand(batch_size, num_docs, 1)
        log_p_pred_y = log_p_pred.gather(2, target_idx).squeeze(2)

        log_marginal = torch.logsumexp(log_p_ret + log_p_pred_y, dim=-1)

        return -log_marginal.mean()


class MarginalizedRetrievalSupervisedLoss(SupervisedLoss):
    """Adapter wiring :class:`MarginalizedRetrievalLoss` to ``ModelOutput`` metadata.

    Expects ``predictions.metadata`` to contain ``per_doc_logits`` and
    ``differentiable_doc_scores`` as produced by
    :class:`medrap.model.RetrievalAugmentedModel` with ``marginalized_retrieval=True``.

    Examples:
        >>> import torch
        >>> from medrap.types import ModelOutput
        >>> pred = ModelOutput(
        ...     logits=torch.zeros(2, 2),
        ...     metadata={
        ...         "per_doc_logits": torch.randn(2, 2, 2),
        ...         "differentiable_doc_scores": torch.randn(2, 2),
        ...     },
        ... )
        >>> loss = MarginalizedRetrievalSupervisedLoss()(pred, torch.tensor([0.0, 1.0]))
        >>> loss.shape
        torch.Size([])
        >>> MarginalizedRetrievalSupervisedLoss()(
        ...     ModelOutput(logits=torch.zeros(2, 2)), torch.tensor([0.0, 1.0])
        ... )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: ModelOutput.metadata must include tensor 'per_doc_logits'

        Missing or invalid metadata:

            >>> MarginalizedRetrievalSupervisedLoss()(
            ...     ModelOutput(logits=torch.zeros(2, 2)), {"x": torch.tensor(1.0)}
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: ...tensor targets...
            >>> MarginalizedRetrievalSupervisedLoss()(
            ...     ModelOutput(logits=torch.zeros(1, 2), metadata={}), torch.tensor([0.0])
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: ...per_doc_logits...
            >>> MarginalizedRetrievalSupervisedLoss()(
            ...     ModelOutput(
            ...         logits=torch.zeros(1, 2),
            ...         metadata={"per_doc_logits": torch.zeros(1, 2, 2)},
            ...     ),
            ...     torch.tensor([0.0]),
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: ...differentiable_doc_scores...
            >>> MarginalizedRetrievalSupervisedLoss()(
            ...     ModelOutput(
            ...         logits=torch.zeros(1, 2),
            ...         metadata={
            ...             "per_doc_logits": torch.zeros(1, 2, 2),
            ...             "differentiable_doc_scores": torch.zeros(1, 2, 3),
            ...         },
            ...     ),
            ...     torch.tensor([0.0]),
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: ...(B, K)...
    """

    def __init__(self) -> None:
        super().__init__()
        self._inner = MarginalizedRetrievalLoss()

    def forward(self, predictions: ModelOutput, targets: TaskTargets) -> Tensor:
        if not isinstance(targets, Tensor):
            raise ValueError("MarginalizedRetrievalSupervisedLoss expects tensor targets")
        meta = predictions.metadata
        per_doc_logits = meta.get("per_doc_logits")
        doc_scores = meta.get("differentiable_doc_scores")
        if not isinstance(per_doc_logits, Tensor):
            raise ValueError("ModelOutput.metadata must include tensor 'per_doc_logits'")
        if not isinstance(doc_scores, Tensor):
            raise ValueError("ModelOutput.metadata must include tensor 'differentiable_doc_scores'")
        if doc_scores.ndim != 2:
            raise ValueError(f"differentiable_doc_scores must be (B, K), got {tuple(doc_scores.shape)}")
        targets_long = targets.long()
        return self._inner(
            per_doc_logits=per_doc_logits,
            doc_scores=doc_scores,
            targets=targets_long,
        )
