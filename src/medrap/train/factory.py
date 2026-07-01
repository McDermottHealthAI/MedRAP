"""Instantiation helpers that wire structured configs into live training objects."""

from typing import Any, cast

import torch  # noqa: F401 — needed in doctests
from hydra_zen import instantiate
from meds_torchdata import MEDSTorchBatch  # noqa: F401 — needed in doctests

from ..model.model import RetrievalAugmentedModel
from .lightning_module import MedRAPSupervisedLightningModule

instantiate_any = cast("Any", instantiate)


def instantiate_training_module(config: Any) -> MedRAPSupervisedLightningModule:
    """Instantiate the configured training wrapper around the plain RAP model.

    Args:
        config: Training config containing the plain RAP model composition under the
            top-level pipeline fields and the supervised wrapper/task under
            ``config.training``.

    Returns:
        MedRAPSupervisedLightningModule: Lightning wrapper whose plain model returns
        logits shaped ``(B, config.training.task.output_dim)`` for a batch of size
        ``B``.

    Examples:
        >>> from medrap.configs import RAPTrainConfig
        >>> module = instantiate_training_module(RAPTrainConfig(output_dir="outputs/demo"))
        >>> module.__class__.__name__
        'MedRAPSupervisedLightningModule'
        >>> module.task.output_dim
        1
        >>> module.loss_fn.__class__.__name__
        'BinaryClassificationLoss'
        >>> batch = MEDSTorchBatch(
        ...     code=torch.LongTensor([[101, 7, 0], [42, 3, 0]]),
        ...     numeric_value=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ...     numeric_value_mask=torch.BoolTensor([[False, False, False], [False, False, False]]),
        ...     time_delta_days=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ...     boolean_value=torch.BoolTensor([True, False]),
        ... )
        >>> out = module.model.forward(batch)
        >>> tuple(out.logits.shape)
        (2, 1)
        >>> targets = module.task.extract_targets(batch)
        >>> module.loss_fn(out, targets).ndim
        0
    """
    plain_model = RetrievalAugmentedModel(
        encoder=instantiate_any(config.encoder),
        query_projector=instantiate_any(config.query_projector),
        retriever=instantiate_any(config.retriever),
        retrieval_encoder=instantiate_any(config.retrieval_encoder),
        fusion=instantiate_any(config.fusion),
        pooling=instantiate_any(config.pooling),
        head=instantiate_any(config.head),
        marginalized_retrieval=bool(getattr(config, "marginalized_retrieval", False)),
        marginalized_score_similarity=str(getattr(config, "marginalized_score_similarity", "dot")),
    )
    task = instantiate_any(config.training.task)
    loss_fn = instantiate_any(config.training.loss)
    return instantiate_any(config.training.module, model=plain_model, task=task, loss_fn=loss_fn)
