"""Lightning callbacks for training diagnostics."""

from collections import defaultdict

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torchmetrics.functional.classification import binary_auroc

from .types import ModelOutput


def _positive_class_probs(logits: Tensor) -> Tensor:
    """Map binary logits to probabilities of the positive class (shape ``(N,)``).

    Examples:
        >>> import torch
        >>> p = _positive_class_probs(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        >>> tuple(p.shape)
        (2,)
        >>> bool(p[0] > p[1])
        True
        >>> p1 = _positive_class_probs(torch.tensor([[0.0], [2.0]]))
        >>> bool(p1[1] > p1[0])
        True
        >>> _positive_class_probs(torch.zeros(3))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Expected logits (N, C)...
        >>> _positive_class_probs(torch.zeros(2, 3))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: ...1 or 2 output dims...
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits (N, C), got {tuple(logits.shape)}")
    if logits.shape[1] == 2:
        return torch.softmax(logits, dim=-1)[:, 1]
    if logits.shape[1] == 1:
        return torch.sigmoid(logits.squeeze(-1))
    raise ValueError(f"Expected 1 or 2 output dims for binary AUROC, got {logits.shape[1]}")


def _resolve_val_dataloader(trainer: pl.Trainer):
    """Resolve the first validation dataloader from a Lightning trainer.

    Examples:
        >>> from types import SimpleNamespace
        >>> class _DM:
        ...     def val_dataloader(self):
        ...         return "from_dm"
        >>> _resolve_val_dataloader(SimpleNamespace(datamodule=_DM(), val_dataloaders="ignored"))
        'from_dm'
        >>> class _DMNone:
        ...     def val_dataloader(self):
        ...         return None
        >>> _resolve_val_dataloader(SimpleNamespace(datamodule=_DMNone(), val_dataloaders=["fb"]))
        'fb'
        >>> _resolve_val_dataloader(SimpleNamespace(datamodule=None, val_dataloaders=[])) is None
        True
        >>> _resolve_val_dataloader(SimpleNamespace(datamodule=None, val_dataloaders=[1]))
        1
        >>> _resolve_val_dataloader(SimpleNamespace(datamodule=None, val_dataloaders="solo"))
        'solo'
        >>> _resolve_val_dataloader(SimpleNamespace(datamodule=None, val_dataloaders=None)) is None
        True
    """
    if getattr(trainer, "datamodule", None) is not None:
        dl = trainer.datamodule.val_dataloader()
        if dl is not None:
            return dl
    val = getattr(trainer, "val_dataloaders", None)
    if val is None:
        return None
    if isinstance(val, list):
        return val[0] if val else None
    return val


class EndOfFitValAUROCCallback(Callback):
    """Compute validation-set AUROC once at the end of ``fit`` and log to loggers.

    Uses a single full pass over the validation dataloader (no per-batch metric
    during training). Intended for WandB summary metrics; also calls
    ``log_metrics`` on other Lightning loggers when supported.

    Supports binary logits shaped ``(N, 1)`` (BCE head) or ``(N, 2)`` (two-class
    head, e.g. marginalized retrieval).

    Examples:
        >>> import torch
        >>> from torch.utils.data import DataLoader
        >>> import lightning.pytorch as pl
        >>> from meds_torchdata import MEDSTorchBatch
        >>> from medrap.task import BinaryClassificationTask
        >>> from medrap.types import ModelOutput
        >>> def _auroc_batch(y: bool) -> MEDSTorchBatch:
        ...     b = MEDSTorchBatch(
        ...         code=torch.LongTensor([[1, 2]]),
        ...         numeric_value=torch.zeros(1, 2, dtype=torch.float32),
        ...         numeric_value_mask=torch.zeros(1, 2, dtype=torch.bool),
        ...         time_delta_days=torch.zeros(1, 2, dtype=torch.float32),
        ...     )
        ...     b.boolean_value = torch.BoolTensor([y])
        ...     return b
        >>> class _ToyBinary(pl.LightningModule):
        ...     def __init__(self) -> None:
        ...         super().__init__()
        ...         self.task = BinaryClassificationTask()
        ...         self.lin = torch.nn.Linear(1, 1)
        ...
        ...     def forward(self, batch: MEDSTorchBatch) -> ModelOutput:
        ...         x = batch.code.float().mean(dim=-1, keepdim=True)
        ...         return ModelOutput(logits=self.lin(x))
        ...
        ...     def training_step(self, batch: MEDSTorchBatch, _batch_idx: int) -> torch.Tensor:
        ...         out = self(batch)
        ...         t = self.task.extract_targets(batch)
        ...         return torch.nn.functional.binary_cross_entropy_with_logits(out.logits.squeeze(1), t)
        ...
        ...     def configure_optimizers(self):
        ...         return torch.optim.SGD(self.parameters(), lr=0.1)
        >>> train_ds = [_auroc_batch(False), _auroc_batch(True)]
        >>> val_ds = [_auroc_batch(False), _auroc_batch(True), _auroc_batch(False), _auroc_batch(True)]
        >>> tr = pl.Trainer(
        ...     max_epochs=1,
        ...     logger=False,
        ...     enable_checkpointing=False,
        ...     enable_progress_bar=False,
        ...     enable_model_summary=False,
        ...     callbacks=[EndOfFitValAUROCCallback()],
        ... )
        >>> tr.fit(
        ...     _ToyBinary(),
        ...     train_dataloaders=DataLoader(train_ds, batch_size=None, shuffle=False),
        ...     val_dataloaders=DataLoader(val_ds, batch_size=None, shuffle=False),
        ... )
        >>> tr.state.finished
        True

    Wrapped :class:`MedRAPSupervisedLightningModule` (forward returns ``ModelOutput``):

        >>> import torch
        >>> from torch.utils.data import DataLoader
        >>> import lightning.pytorch as pl
        >>> from meds_torchdata import MEDSTorchBatch
        >>> from medrap.lightning_module import MedRAPSupervisedLightningModule
        >>> from medrap.task import BinaryClassificationTask
        >>> from medrap.types import ModelOutput
        >>> def _auroc_batch2(y: bool) -> MEDSTorchBatch:
        ...     b = MEDSTorchBatch(
        ...         code=torch.LongTensor([[1, 2]]),
        ...         numeric_value=torch.zeros(1, 2, dtype=torch.float32),
        ...         numeric_value_mask=torch.zeros(1, 2, dtype=torch.bool),
        ...         time_delta_days=torch.zeros(1, 2, dtype=torch.float32),
        ...     )
        ...     b.boolean_value = torch.BoolTensor([y])
        ...     return b
        >>> class _Inner(torch.nn.Module):
        ...     def __init__(self) -> None:
        ...         super().__init__()
        ...         self.lin = torch.nn.Linear(1, 1)
        ...
        ...     def forward(self, batch: MEDSTorchBatch) -> ModelOutput:
        ...         x = batch.code.float().mean(dim=-1, keepdim=True)
        ...         return ModelOutput(logits=self.lin(x))
        >>> wrapped = MedRAPSupervisedLightningModule(model=_Inner(), task=BinaryClassificationTask())
        >>> tr2 = pl.Trainer(
        ...     max_epochs=1,
        ...     logger=False,
        ...     enable_checkpointing=False,
        ...     enable_progress_bar=False,
        ...     enable_model_summary=False,
        ...     callbacks=[EndOfFitValAUROCCallback()],
        ... )
        >>> tr2.fit(
        ...     wrapped,
        ...     train_dataloaders=DataLoader([_auroc_batch2(True)], batch_size=None),
        ...     val_dataloaders=DataLoader([_auroc_batch2(False), _auroc_batch2(True)], batch_size=None),
        ... )
        >>> tr2.state.finished
        True
        >>> from types import SimpleNamespace
        >>> from unittest.mock import patch
        >>> cb = EndOfFitValAUROCCallback()
        >>> cb.on_fit_end(SimpleNamespace(sanity_checking=True), MagicMock())
        >>> cb.on_fit_end(
        ...     SimpleNamespace(sanity_checking=False, datamodule=None, val_dataloaders=None),
        ...     MagicMock(),
        ... )
        >>> empty_b = MEDSTorchBatch(
        ...     code=torch.zeros(0, 2, dtype=torch.long),
        ...     numeric_value=torch.zeros(0, 2, dtype=torch.float32),
        ...     numeric_value_mask=torch.zeros(0, 2, dtype=torch.bool),
        ...     time_delta_days=torch.zeros(0, 2, dtype=torch.float32),
        ... )
        >>> empty_b.boolean_value = torch.zeros(0, dtype=torch.bool)
        >>> plm = MagicMock(spec=pl.LightningModule)
        >>> plm.training = True
        >>> plm.device = torch.device("cpu")
        >>> plm.eval = MagicMock()
        >>> plm.train = MagicMock()
        >>> plm.transfer_batch_to_device = lambda batch, device, dataloader_idx=0: batch
        >>> plm.task = BinaryClassificationTask()
        >>> plm.return_value = ModelOutput(logits=torch.zeros(0, 2))
        >>> cb.on_fit_end(
        ...     SimpleNamespace(
        ...         sanity_checking=False,
        ...         global_step=0,
        ...         loggers=[],
        ...         datamodule=None,
        ...         val_dataloaders=DataLoader([empty_b], batch_size=None),
        ...     ),
        ...     plm,
        ... )
        >>> ev = _Inner()
        >>> _ = ev.eval()
        >>> EndOfFitValAUROCCallback().on_fit_end(
        ...     SimpleNamespace(
        ...         sanity_checking=False,
        ...         global_step=1,
        ...         loggers=[],
        ...         datamodule=None,
        ...         val_dataloaders=DataLoader([_auroc_batch2(False), _auroc_batch2(True)], batch_size=None),
        ...     ),
        ...     ev,
        ... )
        >>> ev.training
        False
        >>> plm2 = MagicMock(spec=pl.LightningModule)
        >>> plm2.training = True
        >>> plm2.device = torch.device("cpu")
        >>> plm2.eval = MagicMock()
        >>> plm2.train = MagicMock()
        >>> plm2.transfer_batch_to_device = lambda batch, device, dataloader_idx=0: batch
        >>> plm2.task = MagicMock()
        >>> plm2.task.extract_targets = MagicMock(return_value={"k": torch.tensor(1.0)})
        >>> plm2.return_value = ModelOutput(logits=torch.tensor([[0.0, 1.0]]))
        >>> cb.on_fit_end(
        ...     SimpleNamespace(
        ...         sanity_checking=False,
        ...         global_step=0,
        ...         loggers=[],
        ...         datamodule=None,
        ...         val_dataloaders=DataLoader([_auroc_batch2(True)], batch_size=None),
        ...     ),
        ...     plm2,
        ... )
        >>> cb.on_fit_end(
        ...     SimpleNamespace(
        ...         sanity_checking=False,
        ...         datamodule=SimpleNamespace(
        ...             val_dataloader=lambda: DataLoader([_auroc_batch2(True)], batch_size=None),
        ...         ),
        ...         val_dataloaders=None,
        ...         global_step=3,
        ...         loggers=[],
        ...     ),
        ...     SimpleNamespace(task=None, training=False, eval=lambda: None),
        ... )
        >>> plm3 = MagicMock(spec=pl.LightningModule)
        >>> plm3.training = True
        >>> plm3.device = torch.device("cpu")
        >>> plm3.eval = MagicMock()
        >>> plm3.train = MagicMock()
        >>> plm3.transfer_batch_to_device = lambda batch, device, dataloader_idx=0: batch
        >>> plm3.task = BinaryClassificationTask()
        >>> plm3.return_value = torch.tensor(0.0)
        >>> cb.on_fit_end(
        ...     SimpleNamespace(
        ...         sanity_checking=False,
        ...         global_step=0,
        ...         loggers=[],
        ...         datamodule=None,
        ...         val_dataloaders=DataLoader([_auroc_batch2(False), _auroc_batch2(True)], batch_size=None),
        ...     ),
        ...     plm3,
        ... )
        >>> log_one = SimpleNamespace(log_metrics=MagicMock())
        >>> cb.on_fit_end(
        ...     SimpleNamespace(
        ...         sanity_checking=False,
        ...         global_step=0,
        ...         loggers=[log_one],
        ...         datamodule=None,
        ...         val_dataloaders=DataLoader([_auroc_batch2(False), _auroc_batch2(False)], batch_size=None),
        ...     ),
        ...     ev,
        ... )
        >>> log_one.log_metrics.called
        False
        >>> log_bad = SimpleNamespace(log_metrics=MagicMock())
        >>> with patch("medrap.callbacks.binary_auroc", side_effect=RuntimeError("fail")):
        ...     EndOfFitValAUROCCallback().on_fit_end(
        ...         SimpleNamespace(
        ...             sanity_checking=False,
        ...             global_step=0,
        ...             loggers=[log_bad],
        ...             datamodule=None,
        ...             val_dataloaders=DataLoader(
        ...                 [_auroc_batch2(False), _auroc_batch2(True)], batch_size=None
        ...             ),
        ...         ),
        ...         ev,
        ...     )
        >>> log_bad.log_metrics.called
        False
        >>> log_nan = SimpleNamespace(log_metrics=MagicMock())
        >>> with patch("medrap.callbacks.binary_auroc", return_value=torch.tensor(float("nan"))):
        ...     EndOfFitValAUROCCallback().on_fit_end(
        ...         SimpleNamespace(
        ...             sanity_checking=False,
        ...             global_step=0,
        ...             loggers=[log_nan],
        ...             datamodule=None,
        ...             val_dataloaders=DataLoader(
        ...                 [_auroc_batch2(False), _auroc_batch2(True)], batch_size=None
        ...             ),
        ...         ),
        ...         ev,
        ...     )
        >>> log_nan.log_metrics.called
        False
        >>> import medrap.callbacks as _cb_mod
        >>> class _StubWB:
        ...     pass
        >>> _saved_wb = _cb_mod.WandbLogger
        >>> _cb_mod.WandbLogger = _StubWB
        >>> metrics_l = SimpleNamespace(log_metrics=MagicMock())
        >>> wb = _StubWB()
        >>> wb.experiment = SimpleNamespace(summary={})
        >>> EndOfFitValAUROCCallback().on_fit_end(
        ...     SimpleNamespace(
        ...         sanity_checking=False,
        ...         global_step=9,
        ...         loggers=[metrics_l, wb],
        ...         datamodule=None,
        ...         val_dataloaders=DataLoader([_auroc_batch2(False), _auroc_batch2(True)], batch_size=None),
        ...     ),
        ...     wrapped,
        ... )
        >>> metrics_l.log_metrics.called
        True
        >>> "final_val_auroc" in wb.experiment.summary
        True
        >>> import tempfile as _tf
        >>> from lightning.pytorch.loggers import WandbLogger as _LightningWandb
        >>> class _WandbNoExperiment(_LightningWandb):
        ...     def log_metrics(self, _metrics, **_kwargs):
        ...         pass
        ...
        ...     @property
        ...     def experiment(self):
        ...         return None
        >>> wnx = _WandbNoExperiment(
        ...     project="medrap-cb-doctest",
        ...     offline=True,
        ...     save_dir=_tf.mkdtemp(),
        ... )
        >>> EndOfFitValAUROCCallback().on_fit_end(
        ...     SimpleNamespace(
        ...         sanity_checking=False,
        ...         global_step=10,
        ...         loggers=[wnx],
        ...         datamodule=None,
        ...         val_dataloaders=DataLoader([_auroc_batch2(False), _auroc_batch2(True)], batch_size=None),
        ...     ),
        ...     wrapped,
        ... )
        >>> wr_idle = MedRAPSupervisedLightningModule(model=_Inner(), task=BinaryClassificationTask())
        >>> _ = wr_idle.eval()
        >>> EndOfFitValAUROCCallback().on_fit_end(
        ...     SimpleNamespace(
        ...         sanity_checking=False,
        ...         global_step=11,
        ...         loggers=[],
        ...         datamodule=None,
        ...         val_dataloaders=DataLoader([_auroc_batch2(False), _auroc_batch2(True)], batch_size=None),
        ...     ),
        ...     wr_idle,
        ... )
        >>> not wr_idle.training
        True
        >>> _cb_mod.WandbLogger = _saved_wb
    """

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        val_loader = _resolve_val_dataloader(trainer)
        if val_loader is None:
            return

        task = getattr(pl_module, "task", None)
        if task is None or not hasattr(task, "extract_targets"):
            return

        device = pl_module.device
        pl_module_was_training = pl_module.training
        pl_module.eval()
        probs_chunks: list[Tensor] = []
        target_chunks: list[Tensor] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = pl_module.transfer_batch_to_device(batch, device, dataloader_idx=0)
                out = pl_module(batch)
                if not isinstance(out, ModelOutput):
                    continue
                targets = task.extract_targets(batch)
                if not isinstance(targets, Tensor):
                    continue
                probs_chunks.append(_positive_class_probs(out.logits).detach().float().cpu())
                target_chunks.append(targets.detach().float().cpu().view(-1))

        if not probs_chunks:
            return

        probs = torch.cat(probs_chunks, dim=0)
        targets = torch.cat(target_chunks, dim=0)
        if probs.shape[0] == 0:
            return

        unique = torch.unique(targets.long())
        if unique.numel() < 2:
            return

        try:
            score = binary_auroc(probs, targets.long())
        except Exception:
            return
        if torch.isnan(score):
            return

        value = float(score.item())
        step = int(getattr(trainer, "global_step", 0))

        for lg in trainer.loggers:
            log_metrics = getattr(lg, "log_metrics", None)
            if callable(log_metrics):
                log_metrics({"final/val_auroc": value}, step=step)
            if isinstance(lg, WandbLogger):
                exp = getattr(lg, "experiment", None)
                if exp is not None:
                    exp.summary["final_val_auroc"] = value

        if pl_module_was_training:
            pl_module.train()
        else:
            pl_module.eval()


class GradientNormCallback(Callback):
    """Log L2 gradient norms per top-level parameter group (e.g. ``model``).

    Metrics are named ``grad_norm/train/{module}`` so W&B groups them under a
    dedicated ``grad_norm`` section rather than the headline ``train`` section.
    Also logs ``grad_norm/train/query_projector`` over all parameters whose
    name contains ``query_projector``.

    Uses :meth:`lightning.pytorch.core.module.LightningModule.log` so values
    reach WandB when a ``WandbLogger`` is configured.

    Args:
        every_n_steps: Log at most once every this many global steps (after backward).

    Examples:
        >>> import torch
        >>> from torch.utils.data import DataLoader
        >>> import lightning.pytorch as pl
        >>> class _DummyGrad(pl.LightningModule):
        ...     def __init__(self) -> None:
        ...         super().__init__()
        ...         self.layer = torch.nn.Linear(2, 2)
        ...
        ...     def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        ...         return self.layer(batch).sum()
        ...
        ...     def configure_optimizers(self):
        ...         return torch.optim.SGD(self.parameters(), lr=0.1)
        >>> tr = pl.Trainer(
        ...     max_epochs=1,
        ...     logger=False,
        ...     enable_checkpointing=False,
        ...     enable_progress_bar=False,
        ...     enable_model_summary=False,
        ...     callbacks=[GradientNormCallback(every_n_steps=1)],
        ... )
        >>> _ = tr.fit(
        ...     _DummyGrad(),
        ...     train_dataloaders=DataLoader(torch.randn(4, 2), batch_size=2),
        ... )
        >>> any(str(k).startswith("grad_norm/train") for k in tr.callback_metrics)
        True
        >>> from types import SimpleNamespace
        >>> class _QPG(pl.LightningModule):
        ...     def __init__(self) -> None:
        ...         super().__init__()
        ...         self.query_projector = torch.nn.Linear(2, 2)
        ...         self.other = torch.nn.Linear(2, 2)
        ...
        ...     def training_step(self, batch, _i):
        ...         return self.query_projector(self.other(batch)).sum()
        ...
        ...     def configure_optimizers(self):
        ...         return torch.optim.SGD(self.parameters(), lr=0.1)
        >>> gmod = _QPG()
        >>> gcb = GradientNormCallback(every_n_steps=50)
        >>> gcalls: list = []
        >>> gmod.log = lambda *a, **k: gcalls.append(a[0])
        >>> gb = torch.randn(2, 2)
        >>> gmod.training_step(gb, 0).backward()
        >>> gcb.on_after_backward(SimpleNamespace(global_step=5), gmod)
        >>> gcalls
        []
        >>> gmod.training_step(gb, 0).backward()
        >>> gcb.on_after_backward(SimpleNamespace(global_step=50), gmod)
        >>> any(str(x).startswith("grad_norm/train") for x in gcalls)
        True
        >>> class _Mix(pl.LightningModule):
        ...     def __init__(self) -> None:
        ...         super().__init__()
        ...         self.query_projector = torch.nn.Linear(2, 2)
        ...         self.frozen = torch.nn.Linear(2, 2)
        ...         for p in self.frozen.parameters():
        ...             p.requires_grad_(False)
        ...
        ...     def training_step(self, batch, _i):
        ...         return self.query_projector(self.frozen(batch)).sum()
        ...
        ...     def configure_optimizers(self):
        ...         return torch.optim.SGD(self.query_projector.parameters(), lr=0.1)
        >>> mx = _Mix()
        >>> gcb2 = GradientNormCallback(every_n_steps=1)
        >>> mxcalls: list = []
        >>> mx.log = lambda *a, **k: mxcalls.append(a[0])
        >>> mx.training_step(torch.randn(2, 2), 0).backward()
        >>> gcb2.on_after_backward(SimpleNamespace(global_step=1), mx)
        >>> any("query_projector" in str(x) for x in mxcalls)
        True
        >>> GradientNormCallback(every_n_steps=0).every_n_steps
        1
    """

    def __init__(self, *, every_n_steps: int = 50) -> None:
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))

    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_step % self.every_n_steps != 0:
            return
        sq_by_group: dict[str, float] = defaultdict(float)
        query_projector_sq = 0.0
        for name, parameter in pl_module.named_parameters():
            grad = parameter.grad
            if grad is None:
                continue
            n = float(grad.detach().norm(2).item())
            group = name.split(".", 1)[0]
            sq_by_group[group] += n * n
            if "query_projector" in name:
                query_projector_sq += n * n
        for group, sq in sorted(sq_by_group.items()):
            pl_module.log(
                f"grad_norm/train/{group}",
                sq**0.5,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
        if query_projector_sq > 0.0:
            qp_norm = query_projector_sq**0.5
            pl_module.log(
                "grad_norm/train/query_projector",
                qp_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            pl_module.log(
                "grad_norm/train/query_projection",
                qp_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
        total_sq = sum(sq_by_group.values(), 0.0)
        pl_module.log(
            "grad_norm/train/total",
            total_sq**0.5,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
