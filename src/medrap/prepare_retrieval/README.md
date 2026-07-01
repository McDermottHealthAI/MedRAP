# medrap.prepare_retrieval

`medrap-prepare-retrieval-dataset` is a single CLI command that takes a
Hugging Face dataset of text documents and produces a static retrieval artifact:
a tokenized, embedded dataset saved to disk with a FAISS nearest-neighbor index.
The artifact is what `HFDatasetRetriever` loads at training and inference time.

## How to run

```bash
medrap-prepare-retrieval-dataset \
  prep.source.path=<hf-dataset-name-or-path> \
  prep.source.split=<split> \
  "prep.document.fields=[<col1>,<col2>]" \
  prep.tokenizer.pretrained_model_name_or_path=<model> \
  prep.embedder.model_name_or_path=<model> \
  prep.output.output_dir=<path/to/output>
```

### Example — MedRAG/textbooks, 10 docs

```bash
medrag-prepare-retrieval-dataset \
  prep.source.path=MedRAG/textbooks \
  prep.source.split=train \
  "prep.document.fields=[title,content]" \
  prep.index.source_id_column=id \
  prep.tokenizer.pretrained_model_name_or_path=sentence-transformers/all-MiniLM-L6-v2 \
  prep.embedder.model_name_or_path=sentence-transformers/all-MiniLM-L6-v2 \
  prep.embedder.device=cpu \
  prep.output.output_dir=./retrieval_dataset \
  prep.num_docs=10
```

### Key options

| Option                               | Default      | Description                                                                                                                     |
| ------------------------------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `prep.document.fields`               | **required** | Ordered list of source columns to concatenate into document text                                                                |
| `prep.document.separator`            | `\n`         | String inserted between rendered field fragments                                                                                |
| `prep.document.include_field_names`  | `false`      | Prefix each field value with `"<field>: "`                                                                                      |
| `prep.index.source_id_column`        | `null`       | Source column to use as `doc_ids`; falls back to row index when `null`                                                          |
| `prep.index.max_length`              | 512          | Tokenizer max sequence length (truncates longer documents)                                                                      |
| `prep.index.tokenization_batch_size` | 256          | Batch size for the tokenization map pass                                                                                        |
| `prep.index.embedding_batch_size`    | 256          | Outer batch size passed to `dataset.map` during embedding                                                                       |
| `prep.index.encode_batch_size`       | `null`       | Per-forward-pass batch size inside `embedder.encode`; defaults to `embedding_batch_size` — lower this to reduce peak GPU memory |
| `prep.index.string_factory`          | `null`       | FAISS `string_factory` for non-flat indexes (e.g. `"IVF256,Flat"`)                                                              |
| `prep.num_docs`                      | `null`       | Randomly subsample this many docs before processing; `null` uses all                                                            |
| `prep.num_docs_seed`                 | 42           | Random seed for subsampling                                                                                                     |
| `prep.embedder.device`               | `cpu`        | Device for embedding (`cpu`, `cuda`, `mps`)                                                                                     |
| `do_overwrite`                       | `false`      | Overwrite an existing output directory                                                                                          |

### Loading from disk instead of Hugging Face Hub

To use a locally saved dataset, switch the source config group:

```bash
medrap-prepare-retrieval-dataset \
  prep/source=load_from_disk \
  prep.source.dataset_path=<path/to/local/dataset> \
  ...
```

## What happens under the hood

### Stage 1 — Document rendering

Each row in the source dataset is passed through `OrderedFieldDocumentRenderer`,
which concatenates the configured `fields` in order, joined by `separator`. The
result is written to the `doc_text` column. If `source_id_column` is set, its
value is copied into `doc_ids`; otherwise the row index is used.

### Stage 2 — Tokenization

`doc_text` is tokenized with the configured `AutoTokenizer` using truncation and
`max_length` padding. Token ids and attention masks are written to `doc_tokens`
and `doc_attention_mask`. These are the inputs loaded by the retrieval encoder
at training time.

### Stage 3 — Embedding

`doc_text` is encoded by the configured `SentenceTransformer` embedder. The
resulting float32 vectors are written to `doc_key_embeddings`. These are the
vectors indexed by FAISS and used to score query-document similarity at
retrieval time.

### Stage 4 — FAISS indexing

A FAISS flat L2 index (or the index described by `string_factory`) is built
over `doc_key_embeddings`. The index is saved as `<index_name>.faiss` alongside
the dataset. The embeddings column is kept in the dataset for payload
access during retrieval.

## Output layout

```
<output_dir>/
  config.yaml            # raw Hydra config as passed
  resolved_config.yaml   # fully resolved config (all interpolations expanded)
  data-00000-of-N.arrow  # tokenized + embedded dataset (Arrow format)
  dataset_info.json      # Hugging Face dataset metadata
  state.json             # Hugging Face dataset state
  retrieval.faiss        # FAISS index over doc_key_embeddings
```

The directory is consumed directly by `HFDatasetRetrieverConfig`:

```yaml
retriever:
  _target_: medrap.model.retrievers.load_hf_dataset_retriever
  dataset_path: <output_dir>
  index_name: retrieval
  k: 4
```
