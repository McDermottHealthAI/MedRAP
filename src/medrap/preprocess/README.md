# medrap.preprocess

`medrap-preprocess` filters a raw MEDS dataset before tensorization: it drops
codes below configurable frequency thresholds (exempting a configurable
sentinel-code regex, e.g. death/admission/discharge events, regardless of
frequency) and drops subjects with too few distinct event timepoints. Writes
the filtered dataset in the same raw-MEDS shape as the input.

Run this once per MEDS dataset and configuration, then feed its output
directory into `meds-torch-data`'s own `MTD_preprocess` command for
tensorization — output from that is reused by `medrap-train`,
`medrap-eval`, and the planned `medrap-retrieve`/`medrap-get-embeddings`/
`medrap-predict-probabilities` commands.
