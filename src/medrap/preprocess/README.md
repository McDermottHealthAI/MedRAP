# medrap.preprocess

Takes a raw MEDS directory and produces a pre-processed, tensorized dataset for
ingestion by the model. Run once per MEDS dataset and configuration; output is
reused by `medrap train`, `medrap retrieve`, `medrap get-embeddings`, and
`medrap predict-probabilities`.
