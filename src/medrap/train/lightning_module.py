"""Supervised PyTorch Lightning wrapper for MedRAP."""

from collections.abc import Callable

import lightning
import torch
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn
from torch.optim import Optimizer
from transformers import get_cosine_schedule_with_warmup

from ..types import ModelOutput
from .losses import BinaryClassificationLoss
from .metrics import binary_auroc_torch, multitask_auroc_torch, positive_class_probs
from .retrieval_logging import model_diagnostic_scalars
from .task import (
    BinaryClassificationTask,
    SupervisedLoss,
    SupervisedTask,
)


class MedRAPSupervisedLightningModule(lightning.LightningModule):
    """Supervised Lightning wrapper around a plain RAP model.

    Args:
        model: Plain PyTorch model returning ``ModelOutput``.
        task: Supervised task object.
        loss_fn: Supervised loss object.
        optimizer: Optimizer factory taking grouped parameter configs.
        lr: Learning rate for the default AdamW optimizer.
        weight_decay: Weight decay for the default AdamW optimizer.
        diagnostics_every_n_steps: Train-step interval for lightweight
            diagnostics. ``0`` disables diagnostics during training.
            Validation/test diagnostics are logged as epoch aggregates when
            this is non-zero.
        validation_auroc: Whether to accumulate detached logits and tensor
            targets during each validation pass and log AUROC when that
            validation loop finishes.
        validation_auroc_log_per_task: Whether to also log per-task AUROC for
            multitask logits shaped ``(B, T)``. The mean over tasks with both
            classes present is always logged for multitask outputs.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        task: SupervisedTask | None = None,
        loss_fn: SupervisedLoss | None = None,
        optimizer: Callable[[list[dict[str, object]]], Optimizer] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        diagnostics_every_n_steps: int = 50,
        validation_auroc: bool = True,
        validation_auroc_log_per_task: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.task = task or BinaryClassificationTask()
        self.loss_fn = loss_fn or BinaryClassificationLoss()
        self.optimizer_factory = optimizer or (
            lambda params: torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        )
        self.warmup_steps = warmup_steps
        self.diagnostics_every_n_steps = max(0, int(diagnostics_every_n_steps))
        self.validation_auroc = validation_auroc
        self.validation_auroc_log_per_task = validation_auroc_log_per_task
        self._validation_auroc_logits: list[Tensor] = []
        self._validation_auroc_targets: list[Tensor] = []

    def forward(self, batch: MEDSTorchBatch) -> ModelOutput:
        """Run the wrapped plain model on a MEDS batch.

        Args:
            batch: Input ``MEDSTorchBatch``.

        Returns:
            ModelOutput: Wrapped model output for a batch of size ``B`` with logits
            shaped ``(B, D)`` where ``D`` is the task output width.

        Examples:
            >>> module = MedRAPSupervisedLightningModule(model=ModelOutputBinaryModel())
            >>> output = module.forward(make_supervised_batch())
            >>> isinstance(output, ModelOutput)
            True
            >>> tuple(output.logits.shape)
            (2, 1)
        """
        return self.model(batch)

    def _no_decay_names(self) -> set[str]:
        no_decay_names: set[str] = set()
        norm_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LayerNorm,
            nn.LocalResponseNorm,
        )
        for module_name, module in self.named_modules():
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                if param_name == "bias" or (isinstance(module, norm_modules) and param_name == "weight"):
                    no_decay_names.add(full_name)
        return no_decay_names

    def _grouped_parameters(self) -> list[dict[str, object]]:
        no_decay_names = self._no_decay_names()
        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []

        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name in no_decay_names or parameter.ndim <= 1:
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

        groups: list[dict[str, object]] = []
        if decay_params:
            groups.append({"params": decay_params})
        if no_decay_params:
            groups.append({"params": no_decay_params, "weight_decay": 0.0})
        return groups

    def _run_supervised_step(self, raw_batch: MEDSTorchBatch, *, stage: str) -> Tensor:
        """Run one supervised forward pass, compute loss, and log diagnostics.

        Examples:
            >>> from medrap.train.losses import MarginalizedRetrievalSupervisedLoss
            >>> from medrap.train.task import MarginalizedBinaryClassificationTask
            >>> class _MargModel(nn.Module):
            ...     def forward(self, batch: MEDSTorchBatch) -> ModelOutput:
            ...         return ModelOutput(
            ...             logits=torch.zeros(2, 2),
            ...             metadata={
            ...                 "differentiable_doc_scores": torch.randn(2, 3),
            ...                 "per_doc_logits": torch.randn(2, 3, 2),
            ...             },
            ...         )
            >>> mm = MedRAPSupervisedLightningModule(
            ...     model=_MargModel(),
            ...     task=MarginalizedBinaryClassificationTask(),
            ...     loss_fn=MarginalizedRetrievalSupervisedLoss(),
            ... )
            >>> log_names: list = []
            >>> mm.log = lambda *a, **k: log_names.append(a[0])
            >>> mm.log_dict = lambda d, *a, **k: log_names.extend(d)
            >>> _ = mm._run_supervised_step(make_supervised_batch(), stage="train")
            >>> any(str(n).startswith("retrieval/train/") for n in log_names)
            True
        """
        predictions = self.forward(raw_batch)
        targets = self.task.extract_targets(raw_batch)
        if isinstance(targets, torch.Tensor):
            targets = targets.to(predictions.logits.device)
        loss = self.loss_fn(predictions, targets)

        batch_size = getattr(raw_batch, "batch_size", None)
        if not isinstance(batch_size, int):
            batch_size = predictions.logits.shape[0]

        is_train = stage == "train"
        self.log(
            f"{stage}/loss",
            loss,
            on_step=is_train,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        for metric_name, metric_value in self.task.metrics(predictions, targets).items():
            self.log(
                f"{stage}/{metric_name}",
                metric_value,
                on_step=is_train,
                on_epoch=True,
                prog_bar=not is_train,
                batch_size=batch_size,
            )
        if self.diagnostics_every_n_steps > 0 and (
            not is_train or self.global_step % self.diagnostics_every_n_steps == 0
        ):
            self.log_dict(
                model_diagnostic_scalars(predictions, raw_batch, stage=stage),
                on_step=is_train,
                on_epoch=not is_train,
                prog_bar=False,
                batch_size=batch_size,
            )
        if stage == "val" and isinstance(targets, Tensor):
            self._collect_validation_auroc_batch(predictions.logits, targets)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Clear validation-loop AUROC buffers before each validation pass."""
        self._validation_auroc_logits.clear()
        self._validation_auroc_targets.clear()

    def on_validation_epoch_end(self) -> None:
        """Log AUROC from logits already produced by the validation loop."""
        if not self.validation_auroc or self.trainer.sanity_checking:
            self._validation_auroc_logits.clear()
            self._validation_auroc_targets.clear()
            return
        if not self._validation_auroc_logits:
            return

        logits = torch.cat(self._validation_auroc_logits, dim=0)
        targets = torch.cat(self._validation_auroc_targets, dim=0)
        self._validation_auroc_logits.clear()
        self._validation_auroc_targets.clear()

        metrics = self._validation_auroc_metrics(targets, logits)
        if metrics:
            self.log_dict(metrics, prog_bar=False, logger=True, sync_dist=False)

    def _collect_validation_auroc_batch(self, logits: Tensor, targets: Tensor) -> None:
        if not self.validation_auroc or self.trainer.sanity_checking:
            return
        self._validation_auroc_logits.append(logits.detach().float())
        self._validation_auroc_targets.append(targets.detach().to(device=logits.device).float())

    def _validation_auroc_metrics(self, targets: Tensor, logits: Tensor) -> dict[str, float]:
        """Compute validation-loop AUROC metrics from accumulated tensors.

        Examples:
            >>> module = MedRAPSupervisedLightningModule(model=ModelOutputBinaryModel())
            >>> module._validation_auroc_metrics(torch.ones(2, 1), torch.zeros(2, 1))
            {}
            >>> module._validation_auroc_metrics(torch.FloatTensor([0, 1]), torch.zeros(2, 3))
            {}
            >>> module._validation_auroc_metrics(torch.FloatTensor([1, 1]), torch.zeros(2, 1))
            {}
        """
        if targets.ndim == 2 and logits.ndim == 2 and targets.shape == logits.shape:
            per_task = multitask_auroc_torch(targets, logits)
            if not per_task:
                return {}
            metrics = {"val/auroc/mean": sum(per_task.values()) / len(per_task)}
            if self.validation_auroc_log_per_task:
                metrics.update({f"val/auroc/task_{task_idx}": value for task_idx, value in per_task.items()})
            return metrics

        try:
            probs = positive_class_probs(logits)
        except ValueError:
            return {}
        score = binary_auroc_torch(targets, probs)
        if score is None or not torch.isfinite(score):
            return {}
        return {"val/auroc": float(score.detach().cpu().item())}

    def training_step(self, batch: MEDSTorchBatch, _batch_idx: int) -> Tensor:
        """Compute the supervised training loss for one batch.

        Args:
            batch: Input ``MEDSTorchBatch`` with ``boolean_value`` targets.
            _batch_idx: Unused batch index required by Lightning.

        Returns:
            Tensor: Scalar training loss with shape ``()``.
        """
        return self._run_supervised_step(batch, stage="train")

    def validation_step(self, batch: MEDSTorchBatch, _batch_idx: int) -> Tensor:
        """Compute the supervised validation loss for one batch.

        Args:
            batch: Input ``MEDSTorchBatch`` with ``boolean_value`` targets.
            _batch_idx: Unused batch index required by Lightning.

        Returns:
            Tensor: Scalar validation loss with shape ``()``.
        """
        return self._run_supervised_step(batch, stage="val")

    def test_step(self, batch: MEDSTorchBatch, _batch_idx: int) -> Tensor:
        """Compute the supervised test loss for one batch.

        Args:
            batch: Input ``MEDSTorchBatch`` with ``boolean_value`` targets.
            _batch_idx: Unused batch index required by Lightning.

        Returns:
            Tensor: Scalar test loss with shape ``()``.
        """
        return self._run_supervised_step(batch, stage="test")

    def predict_step(self, batch: MEDSTorchBatch, batch_idx: int) -> dict[str, Tensor]:
        """Extract retrieval artifacts for one batch.

        Runs the model forward pass and collects per-sample retrieval artifacts
        as CPU tensors. Intended for use with ``trainer.predict()`` to extract
        artifacts across a full dataset split.

        Returns a dict with the following keys (keys whose value would be
        ``None`` are omitted):

        =========  =====================  ========  ==============================
        Key        Shape                  dtype     Present when
        =========  =====================  ========  ==============================
        logits     ``(B, C)``             float32   Always
        targets    ``(B,)``               float32   When batch has labels
        query_embeddings                            Always
                   ``(B, R, D_ret)``      float32
        doc_ids    ``(B, R, K)``          int64     Retriever provides them
        doc_scores ``(B, R, K)``          float32   Retriever provides them
        doc_key_embeddings                          Retriever provides them
                   ``(B, R, K, D_ret)``   float32
        per_doc_logits                              marginalized_retrieval=True
                   ``(B, K, C)``          float32
        differentiable_doc_scores                   marginalized_retrieval=True
                   ``(B, K)``             float32
        =========  =====================  ========  ==============================

        ``InMemoryRetriever`` always provides ``doc_ids``, ``doc_scores``, and
        ``doc_key_embeddings``. ``HFDatasetRetriever`` always provides
        ``doc_ids`` (using ``doc_ids_column`` if configured, otherwise the
        FAISS dataset row indices) and ``doc_scores``; ``doc_key_embeddings``
        is provided only when ``doc_key_embeddings_column`` is set.

        Args:
            batch: Input ``MEDSTorchBatch``.
            batch_idx: Batch index (unused but required by Lightning).

        Returns:
            Dict of CPU tensors with retrieval artifacts for this batch.

        Examples:
            >>> from medrap.model.encoders import MEDSCodeEncoder
            >>> from medrap.model.fusion import ReplaceFusion
            >>> from medrap.model.heads import LinearHead
            >>> from medrap.model.pooling import IdentityPooling
            >>> from medrap.model.query_projection import SequenceMeanQueryProjector
            >>> from medrap.model.retrieval_encoder import MeanPooledRetrievalEncoder
            >>> from medrap.model.retrievers import InMemoryRetriever
            >>> from medrap.model.model import RetrievalAugmentedModel
            >>> model = RetrievalAugmentedModel(
            ...     encoder=MEDSCodeEncoder(),
            ...     query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
            ...     retriever=InMemoryRetriever(
            ...         doc_key_embeddings=torch.FloatTensor([[1, 0, 0, 0], [0, 1, 0, 0]]),
            ...         doc_tokens=torch.LongTensor([[1, 2], [3, 4]]),
            ...         doc_attention_mask=torch.BoolTensor([[True, True], [True, True]]),
            ...     ),
            ...     retrieval_encoder=MeanPooledRetrievalEncoder(vocab_size=8, embedding_dim=2),
            ...     fusion=ReplaceFusion(),
            ...     pooling=IdentityPooling(),
            ...     head=LinearHead(in_dim=2, out_dim=1),
            ... )
            >>> module = MedRAPSupervisedLightningModule(model=model)
            >>> batch = make_supervised_batch()
            >>> result = module.predict_step(batch, batch_idx=0)
            >>> sorted(result.keys())
            ['doc_ids', 'doc_key_embeddings', 'doc_scores', 'logits', 'query_embeddings', 'targets']
            >>> result["logits"].device.type
            'cpu'
            >>> tuple(result["logits"].shape)
            (2, 1)
            >>> tuple(result["query_embeddings"].shape)
            (2, 1, 4)
            >>> tuple(result["doc_ids"].shape)
            (2, 1, 1)
            >>> tuple(result["doc_scores"].shape)
            (2, 1, 1)
            >>> tuple(result["doc_key_embeddings"].shape)
            (2, 1, 1, 4)

            Batch without labels: ``'targets'`` is absent from the result.

            >>> unlabeled = MEDSTorchBatch(
            ...     code=torch.LongTensor([[1, 2, 3]]),
            ...     numeric_value=torch.zeros(1, 3),
            ...     numeric_value_mask=torch.zeros(1, 3, dtype=torch.bool),
            ...     time_delta_days=torch.zeros(1, 3),
            ... )
            >>> "targets" not in module.predict_step(unlabeled, batch_idx=0)
            True

            Marginalized-retrieval metadata is passed through to the result dict.

            >>> class _MargPredModel(nn.Module):
            ...     def forward(self, batch: MEDSTorchBatch) -> ModelOutput:
            ...         return ModelOutput(
            ...             logits=torch.zeros(2, 1),
            ...             metadata={
            ...                 "per_doc_logits": torch.zeros(2, 3, 1),
            ...                 "differentiable_doc_scores": torch.zeros(2, 3),
            ...             },
            ...         )
            >>> marg_module = MedRAPSupervisedLightningModule(model=_MargPredModel())
            >>> marg_result = marg_module.predict_step(make_supervised_batch(), batch_idx=0)
            >>> "per_doc_logits" in marg_result and "differentiable_doc_scores" in marg_result
            True
        """
        predictions = self.forward(batch)
        meta = predictions.metadata

        result: dict[str, Tensor] = {
            "logits": predictions.logits.detach().cpu(),
        }

        try:
            targets = self.task.extract_targets(batch)
            if isinstance(targets, Tensor):
                result["targets"] = targets.detach().cpu()
        except (AttributeError, ValueError):
            pass

        query_out = meta.get("query_output")
        if query_out is not None:
            result["query_embeddings"] = query_out.query_embeddings.detach().cpu()

        retriever_out = meta.get("retriever_output")
        if retriever_out is not None:
            for name in ("doc_ids", "doc_scores", "doc_key_embeddings"):
                value = getattr(retriever_out, name, None)
                if value is not None:
                    result[name] = value.detach().cpu()

        for name in ("per_doc_logits", "differentiable_doc_scores"):
            value = meta.get(name)
            if isinstance(value, Tensor):
                result[name] = value.detach().cpu()

        return result

    def configure_optimizers(self) -> Optimizer | dict[str, object]:
        """Construct the optimizer for the wrapped plain model.

        Returns:
            Configured optimizer with grouped weight decay.

        Examples:
            >>> module = MedRAPSupervisedLightningModule(model=ModelOutputBinaryModel())
            >>> optimizer = module.configure_optimizers()
            >>> isinstance(optimizer, torch.optim.AdamW)
            True
            >>> len(optimizer.param_groups)
            2
            >>> module.model.linear.bias.requires_grad = False
            >>> optimizer = module.configure_optimizers()
            >>> optimized_params = {
            ...     id(parameter) for group in optimizer.param_groups for parameter in group["params"]
            ... }
            >>> id(module.model.linear.bias) in optimized_params
            False
            >>> class LearnableTask(SupervisedTask):
            ...     def __init__(self) -> None:
            ...         super().__init__(output_dim=1)
            ...         self.scale = nn.Parameter(torch.ones(()))
            ...
            ...     def extract_targets(self, batch: MEDSTorchBatch) -> Tensor:
            ...         return batch.boolean_value.float()
            ...
            ...     def metrics(self, predictions: ModelOutput, targets: object) -> dict[str, Tensor]:
            ...         return {}
            >>> class LearnableLoss(SupervisedLoss):
            ...     def __init__(self, task: LearnableTask) -> None:
            ...         super().__init__()
            ...         self.task = task
            ...
            ...     def forward(self, predictions: ModelOutput, targets: object) -> Tensor:
            ...         assert isinstance(targets, Tensor)
            ...         return torch.nn.functional.binary_cross_entropy_with_logits(
            ...             self.task.scale * predictions.logits.squeeze(1),
            ...             targets,
            ...         )
            >>> task = LearnableTask()
            >>> module = MedRAPSupervisedLightningModule(
            ...     model=ModelOutputBinaryModel(),
            ...     task=task,
            ...     loss_fn=LearnableLoss(task),
            ... )
            >>> optimizer = module.configure_optimizers()
            >>> optimized_params = {
            ...     id(parameter) for group in optimizer.param_groups for parameter in group["params"]
            ... }
            >>> id(task.scale) in optimized_params
            True
            >>> from types import SimpleNamespace
            >>> warmup_module = MedRAPSupervisedLightningModule(
            ...     model=ModelOutputBinaryModel(), warmup_steps=10
            ... )
            >>> warmup_module._trainer = SimpleNamespace(estimated_stepping_batches=float("inf"))
            >>> warmup_module.configure_optimizers()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: configure_optimizers: cannot build cosine LR schedule...
        """
        optimizer = self.optimizer_factory(self._grouped_parameters())
        if self.warmup_steps == 0:
            return optimizer
        num_steps = self.trainer.estimated_stepping_batches
        if not isinstance(num_steps, int):
            raise ValueError(
                "configure_optimizers: cannot build cosine LR schedule because "
                "estimated_stepping_batches is not an integer (the dataset may not implement "
                "__len__). Use a dataset with a known length or pass max_steps to the Trainer."
            )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
