"""This module provides loss functions for retrieval-augmented pretraining."""

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


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
