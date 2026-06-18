# medrap.model

PyTorch `nn.Module` building blocks for the retrieval-augmented pipeline. Contains the patient
encoder variants, query projectors, retrievers, retrieval encoders, fusion modules, pooling
modules, prediction heads, the end-to-end `RetrievalAugmentedModel` orchestrator, and the
differentiable retrieval scoring function used during the model's forward pass.

These components are shared across all CLI commands — `train` trains them and `retrieve`,
`get-embeddings`, and `predict-probabilities` load them from a checkpoint.
