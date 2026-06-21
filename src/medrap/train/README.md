# medrap.train

PyTorch Lightning training infrastructure for `medrap train` and `medrap eval`.

| Module                    | Contents                                                                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `lightning_module.py`     | `MedRAPSupervisedLightningModule` — wraps a `RetrievalAugmentedModel` for supervised training                                         |
| `task.py`                 | Task definitions: `BinaryClassificationTask`, `MarginalizedBinaryClassificationTask`, `MultiTaskBinaryClassificationTask`             |
| `losses.py`               | Loss functions: `BinaryClassificationLoss`, `MarginalizedRetrievalSupervisedLoss`, `MultiTaskBCELoss`, `MultiTaskBCEMarginalizedLoss` |
| `callbacks.py`            | `EndOfFitValAUROCCallback` and other training callbacks                                                                               |
| `metrics.py`              | AUROC and other scalar metrics used during training                                                                                   |
| `retrieval_logging.py`    | Batch-level retrieval diagnostics logged to W&B / Lightning                                                                           |
| `datamodule.py`           | `SyntheticSupervisedDatamodule` for smoke-testing without real data                                                                   |
| `multitask_datamodule.py` | `MultiTaskMEDSDatamodule` and `MultiTaskMEDSDataset` for simultaneous multi-label prediction                                          |
