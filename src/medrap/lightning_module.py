"""Supervised PyTorch Lightning wrapper for MedRAP."""

from collections.abc import Callable

import lightning
import torch
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn
from torch.optim import Optimizer
from transformers import get_cosine_schedule_with_warmup

from .retrieval_logging import retrieval_diagnostic_scalars
from .task import (
    BinaryClassificationLoss,
    BinaryClassificationTask,
    SupervisedLoss,
    SupervisedTask,
)
from .types import ModelOutput


class MedRAPSupervisedLightningModule(lightning.LightningModule):
    """Supervised Lightning wrapper around a plain RAP model.

    Args:
        model: Plain PyTorch model returning ``ModelOutput``.
        task: Supervised task object.
        loss_fn: Supervised loss object.
        optimizer: Optimizer factory taking grouped parameter configs.
        lr: Learning rate for the default AdamW optimizer.
        weight_decay: Weight decay for the default AdamW optimizer.
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
    ) -> None:
        super().__init__()
        self.model = model
        self.task = task or BinaryClassificationTask()
        self.loss_fn = loss_fn or BinaryClassificationLoss()
        self.optimizer_factory = optimizer or (
            lambda params: torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        )
        self.warmup_steps = warmup_steps

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

    def _iter_no_decay_names(self) -> set[str]:
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
        no_decay_names = self._iter_no_decay_names()
        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []

        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name in no_decay_names:
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
        if isinstance(targets, Tensor):
            for name, value in retrieval_diagnostic_scalars(predictions, targets).items():
                self.log(
                    f"{stage}/{name}",
                    value,
                    on_step=is_train,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size,
                )
        return loss

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
            >>> from medrap.encoders import MEDSCodeEncoder
            >>> from medrap.fusion import ReplaceFusion
            >>> from medrap.heads import LinearHead
            >>> from medrap.pooling import IdentityPooling
            >>> from medrap.query_projection import SequenceMeanQueryProjector
            >>> from medrap.retrieval_encoder import MeanPooledRetrievalEncoder
            >>> from medrap.retrievers import InMemoryRetriever
            >>> from medrap.model import RetrievalAugmentedModel
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
        except Exception:
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

    def configure_optimizers(self) -> Optimizer:
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
            >>> from medrap.losses import MarginalizedRetrievalSupervisedLoss
            >>> from medrap.task import MarginalizedBinaryClassificationTask
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
            >>> _ = mm._run_supervised_step(make_supervised_batch(), stage="train")
            >>> any(str(n).startswith("train/retrieval/") for n in log_names)
            True
        """
        optimizer = self.optimizer_factory(self._grouped_parameters())
        if self.warmup_steps == 0:
            return optimizer
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
