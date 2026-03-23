"""Lightning datamodules for supervised MedRAP training."""

import lightning
from meds_torchdata import MEDSPytorchDataset, MEDSTorchBatch, MEDSTorchDataConfig
from torch.utils.data import DataLoader


class SyntheticSupervisedDatamodule(lightning.LightningDataModule):
    """Deterministic datamodule for synthetic supervised experiments.

    Args:
        train_batch: Training batch with labels shaped ``(B,)``.
        val_batch: Optional validation batch. Defaults to ``train_batch``.
        test_batch: Optional test batch. Defaults to ``val_batch`` or ``train_batch``.
        train_repeat: Number of training batches to yield per epoch.
        val_repeat: Number of validation batches to yield per epoch.
        test_repeat: Number of test batches to yield per epoch.
    """

    def __init__(
        self,
        *,
        train_batch: MEDSTorchBatch,
        val_batch: MEDSTorchBatch | None = None,
        test_batch: MEDSTorchBatch | None = None,
        train_repeat: int = 2,
        val_repeat: int = 1,
        test_repeat: int = 1,
    ) -> None:
        super().__init__()
        self.train_batch = train_batch
        self.val_batch = val_batch or train_batch
        self.test_batch = test_batch or self.val_batch
        self.train_repeat = train_repeat
        self.val_repeat = val_repeat
        self.test_repeat = test_repeat

    def _loader(self, batch: MEDSTorchBatch, *, repeat: int, shuffle: bool) -> DataLoader:
        """Return a repeated single-batch dataloader.

        Examples:
            >>> datamodule = SyntheticSupervisedDatamodule(
            ...     train_batch=make_supervised_batch(),
            ...     train_repeat=0,
            ... )
            >>> datamodule.train_dataloader()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: repeat must be at least 1
        """
        if repeat < 1:
            raise ValueError("repeat must be at least 1")
        items = [batch] * repeat
        return DataLoader(items, batch_size=None, shuffle=shuffle)

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader.

        Returns:
            DataLoader: Loader yielding ``train_repeat`` ``MEDSTorchBatch`` batches.

        Examples:
            >>> datamodule = SyntheticSupervisedDatamodule(
            ...     train_batch=make_supervised_batch(),
            ...     train_repeat=2,
            ... )
            >>> train_loader = datamodule.train_dataloader()
            >>> first = next(iter(train_loader))
            >>> tuple(first.code.shape)
            (2, 3)
            >>> tuple(first.boolean_value.shape)
            (2,)
        """
        return self._loader(self.train_batch, repeat=self.train_repeat, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader.

        Returns:
            DataLoader: Loader yielding ``val_repeat`` ``MEDSTorchBatch`` batches.

        Examples:
            >>> datamodule = SyntheticSupervisedDatamodule(
            ...     train_batch=make_supervised_batch(),
            ...     val_repeat=1,
            ... )
            >>> val_loader = datamodule.val_dataloader()
            >>> first = next(iter(val_loader))
            >>> tuple(first.code.shape)
            (2, 3)
            >>> first.boolean_value.tolist()
            [True, False]
        """
        return self._loader(self.val_batch, repeat=self.val_repeat, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader.

        Returns:
            DataLoader: Loader yielding ``test_repeat`` ``MEDSTorchBatch`` batches.

        Examples:
            >>> datamodule = SyntheticSupervisedDatamodule(
            ...     train_batch=make_supervised_batch(),
            ...     test_repeat=1,
            ... )
            >>> test_loader = datamodule.test_dataloader()
            >>> first = next(iter(test_loader))
            >>> tuple(first.code.shape)
            (2, 3)
            >>> first.boolean_value.tolist()
            [True, False]
        """
        return self._loader(self.test_batch, repeat=self.test_repeat, shuffle=False)


class MEDSDatamodule(lightning.LightningDataModule):
    """Lightning datamodule backed by a tensorized MEDS cohort.

    Wraps ``meds_torchdata.MEDSPytorchDataset`` for the three standard MEDS
    splits: ``train`` → training, ``tuning`` → validation, ``held_out`` → test.
    Call :meth:`setup` before accessing the dataloaders.

    Args:
        tensorized_cohort_dir: Output directory of ``MTD_preprocess``.
        task_labels_dir: Per-split label parquets from ``create_mimic_task_labels.py``
            (optional; omit for unsupervised use).
        max_seq_len: Maximum number of events per subject.
        batch_size: Subjects per batch.
    """

    def __init__(
        self,
        tensorized_cohort_dir: str,
        *,
        task_labels_dir: str | None = None,
        max_seq_len: int = 512,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self._tensorized_cohort_dir = tensorized_cohort_dir
        self._task_labels_dir = task_labels_dir
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def _make_dataset(self, split: str) -> MEDSPytorchDataset:
        cfg = MEDSTorchDataConfig(
            tensorized_cohort_dir=self._tensorized_cohort_dir,
            task_labels_dir=self._task_labels_dir,
            max_seq_len=self.max_seq_len,
            seq_sampling_strategy="to_end",
        )
        return MEDSPytorchDataset(cfg, split=split)

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self._train_ds = self._make_dataset("train")
            self._val_ds = self._make_dataset("tuning")
        if stage in ("test", None):
            self._test_ds = self._make_dataset("held_out")

    def train_dataloader(self) -> DataLoader:
        """Return a shuffled DataLoader over the ``train`` split.

        Examples:
            >>> dm = MEDSDatamodule("dummy", batch_size=4)
            >>> dm._train_ds = batch_dataset(make_meds_batch(4))
            >>> batch = next(iter(dm.train_dataloader()))
            >>> batch.code.shape
            torch.Size([4, 128])
            >>> batch.boolean_value.dtype
            torch.bool
        """
        return DataLoader(
            self._train_ds, batch_size=self.batch_size, collate_fn=self._train_ds.collate, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        """Return a DataLoader over the ``tuning`` split.

        Examples:
            >>> dm = MEDSDatamodule("dummy", batch_size=4)
            >>> dm._val_ds = batch_dataset(make_meds_batch(4))
            >>> batch = next(iter(dm.val_dataloader()))
            >>> batch.code.shape
            torch.Size([4, 128])
            >>> batch.boolean_value.dtype
            torch.bool
        """
        return DataLoader(self._val_ds, batch_size=self.batch_size, collate_fn=self._val_ds.collate)

    def test_dataloader(self) -> DataLoader:
        """Return a DataLoader over the ``held_out`` split.

        Examples:
            >>> dm = MEDSDatamodule("dummy", batch_size=4)
            >>> dm._test_ds = batch_dataset(make_meds_batch(4))
            >>> batch = next(iter(dm.test_dataloader()))
            >>> batch.code.shape
            torch.Size([4, 128])
            >>> batch.boolean_value.dtype
            torch.bool
        """
        return DataLoader(self._test_ds, batch_size=self.batch_size, collate_fn=self._test_ds.collate)
