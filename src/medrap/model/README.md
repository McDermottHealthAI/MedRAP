# medrap.model

PyTorch `nn.Module` building blocks shared across all CLI commands. `train` fits
them; `retrieve`, `get-embeddings`, and `predict-probabilities` load them from a
checkpoint.

| Module                 | Contents                                                                                                                                                             |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model.py`             | `RetrievalAugmentedModel` — end-to-end orchestrator (`encode → query → retrieve → retrieval-encode → fuse → pool → predict`)                                         |
| `encoders.py`          | `MEDSCodeEncoder`, `TokenEmbeddingEncoder`, `TabularEncoder`, `TimeDeltaRoPEPatientEncoder`                                                                          |
| `query_projection.py`  | `LinearQueryProjector`, `SequenceMeanQueryProjector`                                                                                                                 |
| `retrievers.py`        | `InMemoryRetriever`, `HFDatasetRetriever` (FAISS over a prepared HF dataset)                                                                                         |
| `retrieval_encoder.py` | `TokenFeatureRetrievalEncoder`, `MeanPooledRetrievalEncoder`, `PerDocMeanPooledRetrievalEncoder`, `LinearProjectionRetrievalEncoder`, `KeyEmbeddingRetrievalEncoder` |
| `retrieval_scoring.py` | Differentiable retrieval scoring used during the model's forward pass                                                                                                |
| `fusion.py`            | `ReplaceFusion`, `ConcatFusion`, `PassthroughFusion`, `CrossAttentionFusion`                                                                                         |
| `pooling.py`           | `IdentityPooling`, `MaskedMeanPooling`                                                                                                                               |
| `heads.py`             | `LinearHead`                                                                                                                                                         |
