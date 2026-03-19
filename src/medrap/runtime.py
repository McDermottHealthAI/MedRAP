"""Runtime helpers for building RAP models from composed configs."""

from typing import Any

from .configs import instantiate_model


def build_model_from_cfg(cfg: Any):
    """Build a ``RetrievalAugmentedModel`` from Hydra-style component config.

    Examples:
        >>> from medrap.configs import default_pipeline_config
        >>> model = build_model_from_cfg(default_pipeline_config())
        >>> model.__class__.__name__
        'RetrievalAugmentedModel'
    """
    return instantiate_model(cfg)
