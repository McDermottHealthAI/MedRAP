# medrap

[![Status: WIP](https://img.shields.io/badge/status-WIP-orange)](https://github.com/McDermottHealthAI/MedRAP)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/medrap)](https://pypi.org/project/medrap/)
[![Documentation Status](https://readthedocs.org/projects/MedRAP/badge/?version=latest)](https://medrap.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/McDermottHealthAI/MedRAP/actions/workflows/tests.yaml/badge.svg)](https://github.com/McDermottHealthAI/MedRAP/actions/workflows/tests.yaml)
[![Test Coverage](https://codecov.io/github/McDermottHealthAI/MedRAP/graph/badge.svg)](https://codecov.io/github/McDermottHealthAI/MedRAP)
[![Code Quality](https://github.com/McDermottHealthAI/MedRAP/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/McDermottHealthAI/MedRAP/actions/workflows/code-quality-main.yaml)
[![Contributors](https://img.shields.io/github/contributors/McDermottHealthAI/MedRAP.svg)](https://github.com/McDermottHealthAI/MedRAP/graphs/contributors)
[![Pull Requests](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/McDermottHealthAI/MedRAP/pulls)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Retrieval-augmented pretraining (RAP) for MEDS-style EHR data.

## Status

This is a work-in-progress.

Implemented now:

- a concrete pipeline orchestrator (`RetrievalAugmentedModel`)
- simple concrete stage components for smoke usage and examples
- `MEDSCodeEncoder`, which consumes `batch.code` from MEDS-style batches
- a small end-to-end doctest example in `model.py`
- Hydra config groups under `medrap/conf`
- `medrap train` / `medrap eval` CLI entrypoints

## Quickstart (Synthetic MEDS Batch)

```python
import torch

from medrap.encoders import MEDSCodeEncoder
from medrap.fusion import ReplaceFusion
from medrap.heads import IdentityHead
from medrap.model import RetrievalAugmentedModel
from medrap.pooling import IdentityPooling
from medrap.query_projection import IdentityQueryProjector
from medrap.retrieval_encoder import IdentityRetrievalEncoder
from medrap.retrievers import StaticRetriever
from meds_torchdata import MEDSTorchBatch

model = RetrievalAugmentedModel(
    encoder=MEDSCodeEncoder(),
    query_projector=IdentityQueryProjector(),
    retriever=StaticRetriever(doc_tokens=[[1, 2]], doc_attention_mask=[[1, 1]]),
    retrieval_encoder=IdentityRetrievalEncoder(),
    fusion=ReplaceFusion(),
    pooling=IdentityPooling(),
    head=IdentityHead(),
)

batch = MEDSTorchBatch(
    code=torch.LongTensor([[101, 7, 0], [42, 3, 0]]),
    numeric_value=torch.zeros((2, 3), dtype=torch.float32),
    numeric_value_mask=torch.zeros((2, 3), dtype=torch.bool),
    time_delta_days=torch.zeros((2, 3), dtype=torch.float32),
)
out = model.forward(batch)
print(out.logits)  # tensor([[1, 2]])
```

## MEDS Batch Typing

`MEDSCodeEncoder` accepts `meds_torchdata.MEDSTorchBatch` directly.

## CLI

Run with Hydra overrides:

```bash
uv run medrap train run_smoke=false
uv run medrap eval run_smoke=false
```

`medrap` is a thin dispatcher; `train` and `eval` are implemented as Hydra-native
entrypoints (`@hydra.main`) internally.

Hydra component groups live in:

- `encoder/`
- `query_projector/`
- `retriever/`
- `retrieval_encoder/`
- `fusion/`
- `pooling/`
- `head/`

## Using MIMIC-IV Data

MedRAP works with any data in the
[MEDS format](https://medical-event-data-standard.github.io/). This section walks
through the full pipeline for **MIMIC-IV**: downloading the raw data, converting it
to MEDS, tensorizing it for PyTorch, creating task labels, and training a model.

### Prerequisites

- A [PhysioNet](https://physionet.org/) account with MIMIC-IV credentialed access.
- `uv` (or `pip`) for package management.

### Step 1 — Download and convert to MEDS format

Follow the steps in [MIMIC_IV_MEDS](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS) package README to download mimic data.

This produces the following layout:

```
mimic/
├── raw_input/          # Downloaded raw MIMIC-IV files
├── pre_MEDS/           # Intermediate pre-processing
└── MEDS_cohort/        # Final MEDS dataset
    ├── data/
    │   ├── train/0.parquet
    │   ├── tuning/0.parquet
    │   └── held_out/0.parquet
    └── metadata/
        ├── codes.parquet
        ├── dataset.json
        └── subject_splits.parquet
```

Each shard parquet has columns: `subject_id`, `time`, `code`, `numeric_value`.

### Step 2 — Tensorize for PyTorch

With the ETL venv **deactivated** and the MedRAP project environment in use
(`uv sync`; optionally `source .venv/bin/activate`), run:

`meds-torch-data` (already a MedRAP dependency) converts MEDS parquets into an
efficient on-disk tensor format:

```bash
uv run MTD_preprocess \
        MEDS_dataset_dir=mimic/MEDS_cohort \
        output_dir=mimic/tensorized
```

The output in `mimic/tensorized/` is what `meds_torchdata.MEDSTorchDataConfig`
expects as `tensorized_cohort_dir`.

### Step 3 — Create task labels

MedRAP includes a script to extract binary classification labels from the MEDS
cohort. For example, **in-hospital mortality**:

```bash
uv run python scripts/create_mimic_task_labels.py \
        --meds-dir mimic/MEDS_cohort \
        --output-dir mimic/task_labels/in_hospital_mortality \
        --task in_hospital_mortality
```

This produces one parquet file per split (e.g. `train.parquet`, `tuning.parquet`,
`held_out.parquet`) in the
[MEDS label format](https://medical-event-data-standard.github.io/) with columns:

| Column            | Type           | Description                      |
| ----------------- | -------------- | -------------------------------- |
| `subject_id`      | `Int64`        | Patient identifier               |
| `prediction_time` | `Datetime[μs]` | Time at which prediction is made |
| `boolean_value`   | `Boolean`      | Task label (positive / negative) |
