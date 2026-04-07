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
- `medrap prepare-retrieval-dataset` for building static HF retrieval artifacts

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

Build a local Hugging Face dataset + FAISS index for retrieval (requires the `prep` extra, e.g. `uv pip install "medrap[prep]"`). Example: Hub source, then save under `prep.output.output_dir`:

```bash
uv run medrap prepare-retrieval-dataset \
    prep.source.path=MedRAG/textbooks prep.source.split=train \
    prep.document.fields='[title,content]' \
    prep.tokenizer.pretrained_model_name_or_path=Qwen/Qwen3-Embedding-0.6B \
    prep.embedder.model_name_or_path=Qwen/Qwen3-Embedding-0.6B prep.embedder.device=cuda \
    prep.index.source_id_column=id \
    prep.output.output_dir=outputs/retrieval_artifact \
    prep.index.max_length=256 \
    prep.index.tokenization_batch_size=512 \
    prep.index.embedding_batch_size=256 \
    prep.index.encode_batch_size=32
```

Use `prep.embedder.device=cpu` when no GPU is available. For a dataset already on disk, switch the source group: `prep/source=load_from_disk` and set `prep.source.dataset_path=...`.

`medrap` is a thin dispatcher; `train` and `eval` are implemented as Hydra-native
entrypoints (`@hydra.main`) internally, and `prepare-retrieval-dataset` is the
offline artifact-preparation entrypoint.

Hydra component groups live in:

- `encoder/`
- `query_projector/`
- `retriever/`
- `retrieval_encoder/`
- `fusion/`
- `pooling/`
- `head/`
- `prep/`

## Using MIMIC-IV Data

MedRAP works with any data in the
[MEDS format](https://medical-event-data-standard.github.io/). This section walks
through the full pipeline for **MIMIC-IV**: downloading the raw data, converting it
to MEDS, tensorizing it for PyTorch, creating task labels, and training a model.

### Prerequisites

- A [PhysioNet](https://physionet.org/) account with MIMIC-IV credentialed access.
- `uv` (or `pip`) for package management.

### Optional: use a separate ETL environment

To avoid interfering with the main MedRAP model environment, you can run the
MIMIC-IV download and conversion steps in a dedicated `uv` virtual environment:

```bash
uv venv .venv-mimic
source .venv-mimic/bin/activate
uv pip install MIMIC_IV_MEDS MEDS-DEV
```

After finishing data download/conversion/label extraction, deactivate this
environment and return to the MedRAP project environment for tensorization and
model training:

```bash
deactivate
uv sync
source .venv/bin/activate
```

### Step 1 — Download and convert to MEDS format

Follow the steps in [MIMIC_IV_MEDS](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS) package README to download mimic data.

### Step 2 — Create task labels

Follow the steps in [MEDS-DEV](https://github.com/Medical-Event-Data-Standard/MEDS-DEV?tab=readme-ov-file#extracting-a-task) to extract the task labels. For example,

```bash
meds-dev-task \
  task=mortality/in_icu/first_24h \
  dataset=$DATASET_NAME \
  output_dir=$LABELS_DIR \
  dataset_dir=$MEDS_COHORT_DIR
```

### Step 3 — Tensorize for PyTorch

Use [`meds-torch-data`](https://github.com/mmcdermott/meds-torch-data?tab=readme-ov-file#step-2-data-tensorization) to convert MEDS parquets into an efficient on-disk tensor format:

```bash
uv run MTD_preprocess \
       MEDS_dataset_dir=mimic/MEDS_cohort \
       output_dir=mimic/tensorized
```

The output in `mimic/tensorized/` is what `meds_torchdata.MEDSTorchDataConfig`
expects as `tensorized_cohort_dir`.
