"""Lightning datamodules for supervised MedRAP training."""

import lightning
from meds_torchdata import MEDSTorchBatch
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
        self.train_repeat = int(train_repeat)
        self.val_repeat = int(val_repeat)
        self.test_repeat = int(test_repeat)

    def _loader(self, batch: MEDSTorchBatch, *, repeat: int, shuffle: bool) -> DataLoader:
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
