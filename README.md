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

This is a work-in-progress, but the core research stack is implemented.

MedRAP composes a retrieval-augmented patient model from swappable stages
(`encode → query → retrieve → retrieval-encode → fuse → pool → predict`),
orchestrated by `RetrievalAugmentedModel`. The pipeline also supports
REALM-style **marginalized retrieval** (per-document logits marginalized over
the retrieved set, with differentiable document scores for end-to-end retriever
training) via the `marginalized_retrieval` flag.

Implemented stage components (all in `medrap.model`):

- **Encoders**: `MEDSCodeEncoder`, `TokenEmbeddingEncoder`, and
    `TimeDeltaRoPEPatientEncoder` (transformer encoder with rotary position
    embeddings derived from cumulative log time-deltas).
- **Query projectors**: `LinearQueryProjector`, `SequenceMeanQueryProjector`.
- **Retrievers**: `InMemoryRetriever` and `HFDatasetRetriever`
    (FAISS over a prepared HF dataset, with optional GPU FAISS, payload caching,
    and a `none`/`random_docs` retrieval ablation mode).
- **Retrieval encoders**: `TokenFeatureRetrievalEncoder`,
    `MeanPooledRetrievalEncoder`, `PerDocMeanPooledRetrievalEncoder`,
    `LinearProjectionRetrievalEncoder`, `KeyEmbeddingRetrievalEncoder`.
- **Fusion**: `ReplaceFusion`, `ConcatFusion`, `PassthroughFusion`
    (retrieval-ablation baseline), `CrossAttentionFusion`, `PerDocCrossAttentionFusion`
    (per-document cross-attention; required for `marginalized_retrieval` with attention-based fusion).
- **Pooling**: `IdentityPooling`, `MaskedMeanPooling`.
- **Heads**: `LinearHead`.
- **Tasks & losses** (in `medrap.train`): binary, marginalized-binary, and
    multi-task binary classification, with `BinaryClassificationLoss`,
    `MarginalizedRetrievalSupervisedLoss`, `MultiTaskBCELoss`, and
    `MultiTaskBCEMarginalizedLoss`.

## Package layout

```
src/medrap/
├── model/              # nn.Module building blocks (shared by all commands)
├── train/              # Lightning training infrastructure (train, eval)
├── prepare_retrieval/  # Offline retrieval dataset preparation
├── preprocess/         # Raw-MEDS rare-code/sparse-subject filtering (pre-tensorization)
├── retrieve/           # Batch retrieval from a trained model
├── get_embeddings/     # Embedding extraction from a trained model
└── predict_probabilities/  # Probability prediction from a trained model
```

Each subpackage has a `README.md` describing its contents. `types.py`,
`configs.py`, `cli.py`, and `extraction.py` live at the top level.

## Quickstart (Synthetic MEDS Batch)

Assemble a retrieval-augmented model from concrete stage components and run a
forward pass over a synthetic `MEDSTorchBatch`:

```python
import torch
from meds_torchdata import MEDSTorchBatch

from medrap.model.encoders import MEDSCodeEncoder
from medrap.model.fusion import ReplaceFusion
from medrap.model.heads import LinearHead
from medrap.model.model import RetrievalAugmentedModel
from medrap.model.pooling import IdentityPooling
from medrap.model.query_projection import SequenceMeanQueryProjector
from medrap.model.retrieval_encoder import MeanPooledRetrievalEncoder
from medrap.model.retrievers import InMemoryRetriever

model = RetrievalAugmentedModel(
    encoder=MEDSCodeEncoder(),
    query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
    retriever=InMemoryRetriever(
        doc_key_embeddings=torch.FloatTensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ),
        doc_tokens=torch.LongTensor([[1, 2], [3, 4]]),
        doc_attention_mask=torch.BoolTensor([[True, True], [True, True]]),
    ),
    retrieval_encoder=MeanPooledRetrievalEncoder(vocab_size=8, embedding_dim=2),
    fusion=ReplaceFusion(),
    pooling=IdentityPooling(),
    head=LinearHead(in_dim=2, out_dim=2),
)

batch = MEDSTorchBatch(
    code=torch.LongTensor([[101, 0], [42, 7]]),
    numeric_value=torch.zeros((2, 2), dtype=torch.float32),
    numeric_value_mask=torch.zeros((2, 2), dtype=torch.bool),
    time_delta_days=torch.zeros((2, 2), dtype=torch.float32),
)
out = model.forward(batch=batch)
print(tuple(out.logits.shape))  # (2, 2)
print(sorted(out.metadata))
# ['fusion_output', 'query_output', 'retrieval_encoder_output', 'retriever_output']
```

A maintained, doctested copy of this example (plus a marginalized-retrieval
variant) lives in the `RetrievalAugmentedModel` docstring in `model/model.py`.

## MEDS Batch Typing

`MEDSCodeEncoder` accepts `meds_torchdata.MEDSTorchBatch` directly.

## CLI

`medrap` exposes four Hydra-native entrypoints; all accept Hydra overrides and
`medrap-train`/`medrap-eval`/`medrap-preprocess` require an `output_dir`.

A CPU smoke run on the built-in synthetic datamodule (no external data needed):

```bash
uv run medrap-train output_dir=/tmp/medrap_smoke \
	training/datamodule=synthetic training/trainer=lightning_demo
```

Train and evaluate on real (tensorized MEDS) data:

```bash
uv run medrap-train output_dir=outputs/run_001 training/datamodule=meds
uv run medrap-eval output_dir=outputs/run_001_eval \
	checkpoint_path=outputs/run_001/best_model.ckpt
```

Re-running into an existing `output_dir` is rejected unless you pass
`do_overwrite=true` or `do_resume=true` (train), or choose a fresh `output_dir`
(eval). `medrap-eval` also accepts `eval_mode=validate` (default) or
`eval_mode=test`.

Build a local Hugging Face dataset + FAISS index for retrieval (requires the `prep` extra, e.g. `uv pip install "medrap[prep]"`). Example: Hub source, then save under `prep.output.output_dir`:

```bash
uv run medrap-prepare-retrieval-dataset \
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

Use `prep.embedder.device=cpu` when no GPU is available. For a dataset already on disk, switch the source group: `prep/source=load_from_disk` and set `prep.source.dataset_path=...`. Like train/eval,
re-running into an existing `prep.output.output_dir` is rejected unless you pass `do_overwrite=true`.

Filter a raw MEDS dataset's rare codes and sparse subjects before tensorization (run before `MTD_preprocess`, see [Using MIMIC-IV Data](#using-mimic-iv-data) below):

```bash
uv run medrap-preprocess meds_data_dir=mimic/MEDS_cohort output_dir=mimic/MEDS_cohort_filtered \
	min_subjects_per_code=25 min_events_per_subject=5
```

Codes matching `sentinel_code_regex` (death/admission/discharge/registration events by default) are never
dropped regardless of frequency. Like the other commands, re-running into an existing `output_dir` is
rejected unless you pass `do_overwrite=true`.

These commands are direct Hydra entrypoints (`@hydra.main`), so Hydra receives
overrides without an intermediate subcommand dispatcher.

### Hydra config groups

Pipeline-stage groups (under `src/medrap/conf/`):

- `encoder/` — `meds_code`, `rope`, `token_embedding`, `token_embedding_128`, `token_embedding_1024`
- `query_projector/` — `linear`, `sequence_mean`, `sequence_mean_1024`
- `retriever/` — `in_memory`, `in_memory_sanity`, `in_memory_from_pt`, `hf_dataset`
- `retrieval_encoder/` — `token_feature`, `mean_pooled`, `per_doc_mean_pooled`, `linear_projection`, `key_embedding`
- `fusion/` — `replace`, `concat`, `passthrough`, `cross_attention_medium`, `cross_attention_perdoc_medium`
- `pooling/` — `identity`, `masked_mean`
- `head/` — `linear`, `linear_1024_to_2`

Training groups (under `training/`):

- `training/datamodule/` — `synthetic`, `synthetic_marginalized`, `meds`, `meds_multitask`
- `training/loss/` — `binary_bce`, `marginalized_retrieval`, `multitask_binary_bce`, `multitask_binary_bce_marginalized`
- `training/task/` — `binary_classification`, `marginalized_binary`, `multitask_binary`
- `training/module/` — `supervised_lightning`
- `training/trainer/` — `lightning_default`, `lightning_demo`, `lightning_eval`, `lightning_wandb`

Offline retrieval-prep groups (under `prep/`):

- `prep/source/` — `load_dataset`, `load_from_disk`
- `prep/document/` — `ordered_fields`
- `prep/tokenizer/` — `hf_auto`
- `prep/embedder/` — `sentence_transformer`
- `prep/index/` — `default`
- `prep/output/` — `default`

### HF Retrieval Performance

The default locked dependency is `faiss-cpu`, so HF dataset retrieval runs on CPU unless configured
otherwise. If your environment provides a CUDA-enabled FAISS build, you can request GPU FAISS search:

```bash
retriever=hf_dataset \
	retriever.dataset_path=/path/to/retrieval_db \
	retriever.device=0
```

GPU FAISS is optional and not installed by default because available wheels depend on CUDA, Python, and GPU
architecture. If GPU loading fails, leave `retriever.device=null` for CPU retrieval.

To reduce per-batch Hugging Face row materialization overhead, cache retrieval payload columns as tensors:

```bash
retriever.cache_payloads=true \
	retriever.payload_cache_device=cpu
```

Use `retriever.payload_cache_device=cuda` only when the retrieval payloads fit in GPU memory.

## Docker

Build a local image with BuildKit enabled so `uv` downloads are cached across rebuilds:

```bash
DOCKER_BUILDKIT=1 docker build -t medrap:local .
```

GPU training requires Docker with NVIDIA Container Toolkit / NVIDIA container runtime configured on the host.
Verify GPU access from the container with:

```bash
docker run --rm --gpus all medrap:local python -c "import torch; print(torch.cuda.is_available())"
```

The image is intended to consume prebuilt artifacts. For training, mount:

- a tensorized MEDS cohort directory
- a split task-label directory containing `train.parquet`, `tuning.parquet`, and `held_out.parquet`
- a prepared retrieval dataset directory with `retrieval.faiss`
- an output directory for logs and checkpoints

Quick smoke check:

```bash
docker run --rm medrap:local medrap-train --help
```

Example training invocation:

```bash
docker run --rm --gpus all \
	-v /host/tensorized:/data/tensorized:ro \
	-v /host/task_labels:/data/task_labels:ro \
	-v /host/retrieval_db:/data/retrieval_db:ro \
	-v /host/outputs:/outputs \
	medrap:local \
	medrap-train \
	retriever=hf_dataset \
	retriever.dataset_path=/data/retrieval_db \
	training/datamodule=meds \
	training.datamodule.config.tensorized_cohort_dir=/data/tensorized \
	training.datamodule.config.task_labels_dir=/data/task_labels \
	output_dir=/outputs/run_001
```

The image runs as a non-root user by default. If you need bind-mounted output files to match your current
host user, override the runtime user:

```bash
docker run --rm --user "$(id -u):$(id -g)" ...
```

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
