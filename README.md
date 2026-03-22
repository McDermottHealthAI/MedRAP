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

Use a **separate virtual environment** for the
[MIMIC_IV_MEDS](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS) ETL.
That package pins older `meds-transforms` / `polars` than MedRAP’s `meds-torch-data`;
isolating ETL keeps your main MedRAP `.venv` stable.

**Important — one Python environment for the whole ETL**

If you ever ran **`uv tool install MIMIC-IV-MEDS`**, your shell may still pick
**`~/.local/bin/MEDS_extract-MIMIC_IV`** (the tool) even while
**`.venv-mimic-etl`** is activated. Then logs show **configs from**
`~/.local/share/uv/tools/mimic-iv-meds/...` but **`MEDS_transforms` from**
`.venv-mimic-etl/...` — a **mixed stack** that often fails in `shard_events`
(e.g. `…[0-1190).parquet.lock` not found).

1. **Remove the tool** (recommended once per machine):  
   `uv tool uninstall mimic-iv-meds`  
   (or `uv tool list` and uninstall the MIMIC entry you see.)

2. **Prefer running the module with the venv’s interpreter** (always uses that
   env’s `MIMIC_IV_MEDS` + `meds-transforms`, no PATH guessing):

```bash
cd /path/to/MedRAP

uv venv .venv-mimic-etl
uv pip install --python .venv-mimic-etl/bin/python MIMIC-IV-MEDS
# MIMIC-IV-MEDS pins meds-transforms 0.2.x; that release crashes in shard_events with
# FileNotFoundError on paths like .../[0-100).parquet.lock. Bump to the stack MedRAP uses:
uv pip install --python .venv-mimic-etl/bin/python "meds-transforms>=0.5.3,<0.6"

export DATASET_DOWNLOAD_USERNAME=$PHYSIONET_USERNAME
export DATASET_DOWNLOAD_PASSWORD=$PHYSIONET_PASSWORD

# Full dataset (requires ~165 GB memory, ~7 hours)
.venv-mimic-etl/bin/python -m MIMIC_IV_MEDS root_output_dir=mimic/ do_copy=True

# Demo dataset (recommended first)
.venv-mimic-etl/bin/python -m MIMIC_IV_MEDS root_output_dir=mimic/ do_demo=True do_copy=True
```

Optional sanity check after install:

```bash
.venv-mimic-etl/bin/python -c "import MIMIC_IV_MEDS, MEDS_transforms; print(MIMIC_IV_MEDS.__file__); print(MEDS_transforms.__file__)"
# Both paths must be under .../MedRAP/.venv-mimic-etl/...
```

If you use **`source .venv-mimic-etl/bin/activate`** and the **`MEDS_extract-MIMIC_IV`**
script instead, verify **`which MEDS_extract-MIMIC_IV`** points under
**`.venv-mimic-etl/bin`**, not **`~/.local/bin`**.

When finished with ETL, **`deactivate`** if you had activated the venv.

**2. Remove the ETL venv** (optional, saves disk once you do not need to re-run extract):

```bash
rm -rf .venv-mimic-etl
```

**3. Use the main MedRAP environment** for tensorizing, labels, and training only:

```bash
uv sync
# Steps 2+ below: use `uv run …` so commands use `.venv`, not the removed ETL env
```

Before Step 2, confirm the cohort exists, e.g. `ls mimic/MEDS_cohort/data/train/*.parquet`.

**Troubleshooting**

- **`MTD_preprocess` → “No shards found in …/MEDS_cohort/data”`**: the MEDS extract
  pipeline never produced `data/train/*.parquet`. Very often **`mimic/pre_MEDS/.done` is
  left over from a failed or partial run**: with `do_overwrite=False`, `MIMIC-IV-MEDS`
  treats pre-MEDS as done and **`exit(0)` before `MEDS_transform-runner` runs**, so only
  download (and maybe pre-MEDS) happens. **Fix:** remove the marker and partial cohort,
  then re-run extract with the **venv interpreter** (same as Step 1):
  ```bash
  rm -f mimic/pre_MEDS/.done
  rm -rf mimic/MEDS_cohort
  .venv-mimic-etl/bin/python -m MIMIC_IV_MEDS root_output_dir=mimic/ do_demo=True do_copy=True
  ```
  Alternatively pass **`do_overwrite=True`** once (re-runs pre-MEDS; slower). If you
  prefer a clean tree: `rm -rf mimic/` and run extract again.

- **`FileNotFoundError: …/[0-…).parquet.lock` during `shard_events`**: the traceback
  will show **`MEDS_transforms/mapreduce/utils.py`** and a bare **`lock_fp.unlink()`**
  — that is **`meds-transforms` 0.2.x** (what `MIMIC-IV-MEDS` pulls by default), including
  on a **single clean venv** (e.g. cluster clone of `MIMIC_IV_MEDS` with `.venv`). **Fix:**
  upgrade in that same env:  
  `uv pip install "meds-transforms>=0.5.3,<0.6"`  
  (or `pip install …` with the venv activated). Then remove partial output and re-run:
  `rm -rf mimic_data/MEDS_cohort` (or your `root_output_dir`’s `MEDS_cohort`).  
  A **mixed install** (`MEDS_extract-MIMIC_IV` from **`uv tool`** + `MEDS_transforms` from
  the venv) can cause similar failures; use **`uv tool uninstall mimic-iv-meds`** and
  **`python -m MIMIC_IV_MEDS …`** from one env only.

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

| Column            | Type           | Description                        |
|-------------------|----------------|------------------------------------|
| `subject_id`      | `Int64`        | Patient identifier                 |
| `prediction_time` | `Datetime[μs]` | Time at which prediction is made   |
| `boolean_value`   | `Boolean`      | Task label (positive / negative)   |

### Step 4 — Load as MEDSTorchBatch

The existing MedRAP `meds` datamodule (backed by `meds_torchdata`) reads the
tensorized data and task labels natively:

```python
from meds_torchdata import MEDSPytorchDataset, MEDSTorchDataConfig
from torch.utils.data import DataLoader

cfg = MEDSTorchDataConfig(
    tensorized_cohort_dir="mimic/tensorized",
    max_seq_len=128,
    task_labels_dir="mimic/task_labels/in_hospital_mortality",
    seq_sampling_strategy="to_end",
)
dataset = MEDSPytorchDataset(cfg, split="train")
loader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate)

batch = next(iter(loader))  # MEDSTorchBatch
print(batch.code.shape)           # (32, 128)
print(batch.boolean_value.shape)  # (32,)
```

### Step 5 — Train with MedRAP

Use the `medrap train` CLI with Hydra overrides pointing to your data:

```bash
uv run medrap train \
    output_dir=outputs/mimic_mortality \
    training/datamodule=meds \
    training.datamodule.config.tensorized_cohort_dir=mimic/tensorized \
    training.datamodule.config.task_labels_dir=mimic/task_labels/in_hospital_mortality \
    training.datamodule.config.max_seq_len=128 \
    training.datamodule.config.seq_sampling_strategy=to_end
```

This uses the default model components (encoder, retriever, head, etc.) which can
each be swapped via their respective Hydra config groups.

### Project layout reference

After completing all steps, your directory should look like:

```
mimic/
├── MEDS_cohort/           # Step 1 output (MEDS format)
│   ├── data/{split}/*.parquet
│   └── metadata/
├── tensorized/            # Step 2 output (PyTorch tensors)
│   ├── data/{split}/*.nrt
│   └── tokenization/
└── task_labels/           # Step 3 output (MEDS labels)
    └── in_hospital_mortality/
        ├── train.parquet
        ├── tuning.parquet
        └── held_out.parquet
```
