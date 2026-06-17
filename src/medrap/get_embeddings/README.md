# medrap.get_embeddings

Takes a trained model, a retrieval dataset, and an index dataframe for a
pre-processed MEDS dataset. Produces embeddings for each patient-timepoint —
either query embeddings from the encoder/projector or final hidden-layer
representations with inference-style top-document retrieval.

A different retrieval dataset may be used provided the same embedding model was
used to build it. The pre-processed MEDS dataset must match the one used during
training.
