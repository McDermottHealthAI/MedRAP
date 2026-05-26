"""Small tensor metrics used by training diagnostics."""

import torch
from torch import Tensor


def positive_class_probs(logits: Tensor) -> Tensor:
    """Return positive-class probabilities from binary logits.

    Args:
        logits: Binary logits shaped ``(N, 1)`` for BCE-style heads or ``(N, 2)``
            for two-class marginal heads.

    Returns:
        Tensor: Positive-class probabilities with shape ``(N,)``.

    Examples:
        >>> positive_class_probs(torch.FloatTensor([[0.0], [2.0]])).round(decimals=4)
        tensor([0.5000, 0.8808])
        >>> positive_class_probs(torch.FloatTensor([[2.0, 0.0], [0.0, 2.0]])).round(decimals=4)
        tensor([0.1192, 0.8808])
        >>> positive_class_probs(torch.FloatTensor([0.0, 1.0]))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Expected logits shaped (N, C), got (2,)
        >>> positive_class_probs(torch.zeros(2, 3))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Expected 1 or 2 output dims...
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits shaped (N, C), got {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        return torch.sigmoid(logits.squeeze(1))
    if logits.shape[1] == 2:
        return torch.softmax(logits, dim=-1)[:, 1]
    raise ValueError(f"Expected 1 or 2 output dims for binary AUROC, got {logits.shape[1]}")


def binary_auroc_torch(targets: Tensor, scores: Tensor) -> Tensor | None:
    """Compute binary AUROC on a tensor device.

    Args:
        targets: Binary labels. Values greater than ``0.5`` are positive.
        scores: Continuous positive-class scores.

    Returns:
        Tensor | None: Scalar AUROC, or ``None`` when AUROC is undefined because
        only one class is present after finite-value filtering.

    Examples:
        >>> y = torch.FloatTensor([0, 1, 0, 1])
        >>> s = torch.FloatTensor([0.1, 0.9, 0.2, 0.8])
        >>> float(binary_auroc_torch(y, s))
        1.0
        >>> binary_auroc_torch(torch.FloatTensor([1, 1]), torch.FloatTensor([0.2, 0.8])) is None
        True
        >>> binary_auroc_torch(torch.FloatTensor([float("nan"), 1]), torch.FloatTensor([0.2, 0.8])) is None
        True
        >>> float(binary_auroc_torch(torch.FloatTensor([0, 1]), torch.FloatTensor([0.5, 0.5])))
        0.5
    """
    y = targets.reshape(-1).float()
    s = scores.reshape(-1).float()
    valid = torch.isfinite(y) & torch.isfinite(s)
    y = y[valid]
    s = s[valid]
    if y.numel() < 2:
        return None

    y = (y > 0.5).float()
    n_pos = y.sum()
    n_total = torch.as_tensor(y.numel(), device=y.device, dtype=y.dtype)
    n_neg = n_total - n_pos
    if n_pos.item() == 0 or n_neg.item() == 0:
        return None

    sorted_scores, order = torch.sort(s, descending=False)
    sorted_y = y[order]
    n = sorted_scores.numel()

    starts = torch.ones(n, dtype=torch.bool, device=sorted_scores.device)
    starts[1:] = sorted_scores[1:] != sorted_scores[:-1]
    group_starts = torch.nonzero(starts, as_tuple=False).flatten()
    group_ends = torch.cat(
        [
            group_starts[1:],
            torch.tensor([n], dtype=group_starts.dtype, device=group_starts.device),
        ]
    )
    group_ids = torch.cumsum(starts.long(), dim=0) - 1
    pos_by_group = torch.zeros(group_starts.numel(), dtype=torch.float32, device=sorted_scores.device)
    pos_by_group.scatter_add_(0, group_ids, sorted_y)

    avg_ranks = (group_starts.float() + 1.0 + group_ends.float()) / 2.0
    rank_sum_pos = (pos_by_group * avg_ranks).sum()
    return (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def multitask_win_rate(aurocs_a: dict[int, float], aurocs_b: dict[int, float]) -> float | None:
    """Compute the win rate of model A over model B across shared tasks.

    For each task present in both dicts, model A "wins" if its AUROC exceeds
    model B's. The win rate is the fraction of shared tasks won by A. Tasks
    present in only one dict are ignored.

    Args:
        aurocs_a: Per-task AUROC for model A, keyed by task index.
        aurocs_b: Per-task AUROC for model B, keyed by task index.

    Returns:
        float | None: Fraction of shared tasks where A beats B, or ``None``
        when there are no shared tasks.

    Examples:
        >>> multitask_win_rate({0: 0.8, 1: 0.6, 2: 0.7}, {0: 0.75, 1: 0.65, 2: 0.65})
        0.6666666666666666
        >>> multitask_win_rate({0: 0.8, 1: 0.6}, {0: 0.75, 1: 0.65}) == 0.5
        True
        >>> multitask_win_rate({0: 0.8}, {1: 0.7}) is None
        True
        >>> multitask_win_rate({}, {0: 0.7}) is None
        True
    """
    shared = set(aurocs_a) & set(aurocs_b)
    if not shared:
        return None
    wins = sum(1 for t in shared if aurocs_a[t] > aurocs_b[t])
    return wins / len(shared)


def multitask_auroc_torch(targets: Tensor, logits: Tensor) -> dict[int, float]:
    """Compute per-task AUROC for independent binary logits.

    Args:
        targets: Binary task labels shaped ``(N, T)``.
        logits: Per-task logits shaped ``(N, T)``.

    Returns:
        dict[int, float]: AUROC values keyed by task index. Tasks with only one
        observed class are omitted.

    Examples:
        >>> y = torch.FloatTensor([[0, 1], [1, 0], [0, 0], [1, 1]])
        >>> logits = torch.FloatTensor([[-2, 2], [2, -2], [-1, -1], [1, 1]])
        >>> multitask_auroc_torch(y, logits)
        {0: 1.0, 1: 1.0}
        >>> multitask_auroc_torch(y[:, :1], logits)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Expected multitask targets and logits...
    """
    if targets.ndim != 2 or logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError(
            "Expected multitask targets and logits with matching shape (N, T); "
            f"got targets={tuple(targets.shape)}, logits={tuple(logits.shape)}"
        )

    probs = torch.sigmoid(logits.float())
    aurocs: dict[int, float] = {}
    for task_idx in range(probs.shape[1]):
        score = binary_auroc_torch(targets[:, task_idx], probs[:, task_idx])
        if score is not None and torch.isfinite(score):
            aurocs[task_idx] = float(score.detach().cpu().item())
    return aurocs
