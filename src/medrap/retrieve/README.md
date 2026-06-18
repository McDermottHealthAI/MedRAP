# medrap.retrieve

Takes a trained model, a retrieval dataset, and an index dataframe for a
pre-processed MEDS dataset. For each patient-timepoint in the index, retrieves
the top-K documents (or null) and saves document IDs and scores to disk.

A different retrieval dataset may be used provided the same embedding model was
used to build it. The pre-processed MEDS dataset must match the one used during
training.
