"""Verify gender labels are correctly aligned with retrieval artifacts."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from omegaconf import OmegaConf  # noqa: E402

from medrap.configs import instantiate_datamodule  # noqa: E402
from medrap.demographic_analysis import (  # noqa: E402
    build_patient_demographic_frame,
    extract_val_schema,
    load_subject_demographics,
)

# Load artifacts
a = torch.load("outputs/mimic_run_retrieval_only/extraction/extraction_artifacts.pt", weights_only=True)
doc_ids = a["doc_ids"].numpy()
if doc_ids.ndim == 3:
    doc_ids = doc_ids[:, 0, :]

# Load config and datamodule same way as run_demographic_heatmap.py
run_dir = Path("outputs/mimic_run_retrieval_only")
cfg = OmegaConf.load(run_dir / "config.yaml")
datamodule = instantiate_datamodule(cfg)

val_schema = extract_val_schema(datamodule)
demographics = load_subject_demographics(
    "/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort",
    val_schema["subject_id"].to_list(),
)
patient_frame = build_patient_demographic_frame(val_schema, demographics)

print(f"Patient frame rows: {patient_frame.height}")
print(f"Artifact rows: {doc_ids.shape[0]}")
print("\nGender distribution:")
print(patient_frame["gender"].value_counts())

# Check: do M and F patients retrieve different docs?
genders = patient_frame["gender"].to_list()
m_mask = np.array([g == "M" for g in genders])
f_mask = np.array([g == "F" for g in genders])
m_docs = doc_ids[m_mask]
f_docs = doc_ids[f_mask]

m_top = Counter(m_docs.reshape(-1).tolist()).most_common(5)
f_top = Counter(f_docs.reshape(-1).tolist()).most_common(5)
print(f"\nTop-5 docs for M ({m_docs.shape[0]} patients):")
for did, cnt in m_top:
    print(f"  doc_id={did}: {cnt} ({100 * cnt / (m_docs.shape[0] * m_docs.shape[1]):.1f}%)")
print(f"\nTop-5 docs for F ({f_docs.shape[0]} patients):")
for did, cnt in f_top:
    print(f"  doc_id={did}: {cnt} ({100 * cnt / (f_docs.shape[0] * f_docs.shape[1]):.1f}%)")

# Spot-check: first 5 F patients
print("\n--- First 5 F patients ---")
f_sids = []
for i, g in enumerate(genders):
    if g == "F":
        sid = int(patient_frame["subject_id"][i])
        f_sids.append(sid)
        print(f"  row={i}, subject_id={sid}, docs={doc_ids[i]}")
        if len(f_sids) >= 5:
            break

# Verify gender from raw MEDS
print("\n--- Verify gender from raw MEDS for those 5 F patients ---")
cohort = Path("/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort")
raw = (
    pl.scan_parquet(str(cohort / "data" / "*" / "*.parquet"))
    .filter(pl.col("subject_id").is_in(f_sids) & pl.col("code").str.starts_with("GENDER"))
    .select(["subject_id", "code"])
    .collect()
)
print(raw)
