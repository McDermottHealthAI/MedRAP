from medrap.configs import (
    MEDSTorchDataConfigConfig,
    MEDSTrainingDatamoduleConfig,
    RAPTrainConfig,
    TrainingConfig,
    instantiate_datamodule,
)


def test_meds_datamodule_config_instantiates_without_loading_dataset(tmp_path) -> None:
    cohort_dir = tmp_path / "tensorized"
    labels_dir = tmp_path / "labels"
    cohort_dir.mkdir()
    labels_dir.mkdir()

    cfg = RAPTrainConfig(
        output_dir=str(tmp_path / "outputs"),
        training=TrainingConfig(
            datamodule=MEDSTrainingDatamoduleConfig(
                config=MEDSTorchDataConfigConfig(
                    tensorized_cohort_dir=str(cohort_dir),
                    max_seq_len=8,
                    task_labels_dir=str(labels_dir),
                )
            )
        ),
    )

    datamodule = instantiate_datamodule(cfg)

    assert datamodule.__class__.__name__ == "Datamodule"
