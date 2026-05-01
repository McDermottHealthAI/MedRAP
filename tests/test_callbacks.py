from pathlib import Path
from types import SimpleNamespace

from medrap.callbacks import ProgressCheckpointCallback


def test_progress_checkpoint_callback_saves_fractional_milestones(tmp_path: Path) -> None:
    saved: list[Path] = []
    callback = ProgressCheckpointCallback(
        dirpath=tmp_path,
        fractions=[0.25, 0.5, 1.0],
    )
    trainer = SimpleNamespace(
        estimated_stepping_batches=4,
        global_step=0,
        save_checkpoint=lambda path: saved.append(Path(path)),
    )

    callback.on_train_start(trainer, SimpleNamespace())
    for step in range(1, 5):
        trainer.global_step = step
        callback.on_train_batch_end(trainer, SimpleNamespace(), None, None, step)

    assert [path.name for path in saved] == [
        "progress=0.25-step=1.ckpt",
        "progress=0.50-step=2.ckpt",
        "progress=1.00-step=4.ckpt",
    ]


def test_progress_checkpoint_callback_skips_unknown_training_length(tmp_path: Path) -> None:
    saved: list[Path] = []
    callback = ProgressCheckpointCallback(dirpath=tmp_path, fractions=[0.5])
    trainer = SimpleNamespace(
        estimated_stepping_batches=float("inf"),
        global_step=10,
        save_checkpoint=lambda path: saved.append(Path(path)),
    )

    callback.on_train_start(trainer, SimpleNamespace())
    callback.on_train_batch_end(trainer, SimpleNamespace(), None, None, 0)

    assert saved == []
