"""Tests for the Charlson comorbidity lookup and per-patient assignment.

Per AGENTS.md TDD these exercise the public surface that
``src/medrap/comorbidity.py`` must provide for the canonical 17-category
Charlson Comorbidity Index (Charlson 1987, Quan 2005 ICD mapping).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Lookup loading
# ---------------------------------------------------------------------------


def test_charlson_categories_contains_17() -> None:
    """The Charlson taxonomy is canonically 17 categories."""
    from medrap.comorbidity import CHARLSON_CATEGORIES

    assert len(CHARLSON_CATEGORIES) == 17
    for required in (
        "Myocardial infarction",
        "Congestive heart failure",
        "Cerebrovascular disease",
        "Dementia",
        "Chronic pulmonary disease",
        "Diabetes without chronic complications",
        "Diabetes with chronic complications",
        "Renal disease",
        "Any malignancy",
        "Metastatic solid tumor",
        "AIDS/HIV",
    ):
        assert required in CHARLSON_CATEGORIES, required


def test_load_charlson_lookup_recognizes_known_icd10_codes() -> None:
    """Spot-checks for a few well-known ICD-10 codes against Quan 2005."""
    from medrap.comorbidity import load_charlson_lookup, lookup_categories

    lookup = load_charlson_lookup()
    assert isinstance(lookup, dict)
    sample_key = next(iter(lookup))
    assert isinstance(sample_key, tuple) and len(sample_key) == 2

    # CHF: I50, plus a prefix-match check (I5023 starts with I50).
    assert "Congestive heart failure" in lookup_categories(lookup, 10, "I50")
    assert "Congestive heart failure" in lookup_categories(lookup, 10, "I5023")
    # MI: I21.
    assert "Myocardial infarction" in lookup_categories(lookup, 10, "I21")
    # CVD: I63 (cerebral infarction).
    assert "Cerebrovascular disease" in lookup_categories(lookup, 10, "I63")
    # Renal: N18 (CKD).
    assert "Renal disease" in lookup_categories(lookup, 10, "N18")
    # Diabetes without complications: E110 prefix matches.
    assert "Diabetes without chronic complications" in lookup_categories(lookup, 10, "E110")
    # Diabetes with complications: E114.
    assert "Diabetes with chronic complications" in lookup_categories(lookup, 10, "E114")
    # Cancer (non-metastatic): C50 (breast).
    assert "Any malignancy" in lookup_categories(lookup, 10, "C50")
    # Metastatic tumor: C78.
    assert "Metastatic solid tumor" in lookup_categories(lookup, 10, "C78")
    # AIDS: B20.
    assert "AIDS/HIV" in lookup_categories(lookup, 10, "B20")


def test_load_charlson_lookup_recognizes_known_icd9_codes() -> None:
    """ICD-9 prefix-match (decimals stripped)."""
    from medrap.comorbidity import load_charlson_lookup, lookup_categories

    lookup = load_charlson_lookup()
    # CHF: 428 (with decimals stripped). 4280 startswith 428.
    assert "Congestive heart failure" in lookup_categories(lookup, 9, "428")
    assert "Congestive heart failure" in lookup_categories(lookup, 9, "4280")
    # MI: 410.
    assert "Myocardial infarction" in lookup_categories(lookup, 9, "410")
    # CVD: 434.
    assert "Cerebrovascular disease" in lookup_categories(lookup, 9, "434")


def test_load_charlson_lookup_unknown_icd_returns_empty() -> None:
    from medrap.comorbidity import load_charlson_lookup, lookup_categories

    lookup = load_charlson_lookup()
    # I10 (hypertension) is NOT in Charlson — important conceptual difference
    # from Elixhauser.
    assert lookup_categories(lookup, 10, "I10") == frozenset()
    # L20 (atopic dermatitis) — also not Charlson.
    assert lookup_categories(lookup, 10, "L20") == frozenset()
    assert lookup_categories(lookup, 10, "FAKE99") == frozenset()


# ---------------------------------------------------------------------------
# Patient assignment from a synthetic MEDS cohort
# ---------------------------------------------------------------------------


def _write_synthetic_meds_shard(shard_path: Path, events: list[tuple[int, datetime, str]]) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "subject_id": [e[0] for e in events],
            "time": [e[1] for e in events],
            "code": [e[2] for e in events],
        }
    )
    df.write_parquet(shard_path)


def test_assign_patient_charlson_multi_membership_and_filter_by_pred_time(
    tmp_path: Path,
) -> None:
    """One patient with two pre-prediction Charlson diagnoses gets both flags; a post-prediction diagnosis is
    excluded (no label leakage)."""
    from medrap.comorbidity import (
        CHARLSON_CATEGORIES,
        assign_patient_charlson,
        load_charlson_lookup,
    )

    cohort = tmp_path / "MEDS_cohort"
    _write_synthetic_meds_shard(
        cohort / "data" / "tuning" / "shard0.parquet",
        events=[
            # Patient 100: CHF (I50) day 0, DM uncomp (E110) day 1, prediction day 5
            # plus a post-prediction CVD (I63) on day 10 that must NOT contribute.
            (100, datetime(2020, 1, 1), "DIAGNOSIS//ICD//10//I50"),
            (100, datetime(2020, 1, 2), "DIAGNOSIS//ICD//10//E110"),
            (100, datetime(2020, 1, 10), "DIAGNOSIS//ICD//10//I63"),  # post-pred
            # Patient 200: only post-prediction CHF — no flags expected
            (200, datetime(2020, 2, 10), "DIAGNOSIS//ICD//10//I50"),
            # Patient 300: I10 (hypertension) — not in Charlson, no flags
            (300, datetime(2020, 1, 1), "DIAGNOSIS//ICD//10//I10"),
        ],
    )
    val_schema = pl.DataFrame(
        {
            "subject_id": [100, 200, 300],
            "prediction_time": [
                datetime(2020, 1, 5),
                datetime(2020, 2, 5),
                datetime(2020, 1, 5),
            ],
        }
    )

    lookup = load_charlson_lookup()
    result = assign_patient_charlson(cohort, val_schema, lookup=lookup)

    assert result["subject_id"].to_list() == [100, 200, 300]
    for cat in CHARLSON_CATEGORIES:
        assert cat in result.columns, cat
    assert "any_charlson" in result.columns
    assert "n_categories" in result.columns

    # Patient 100: CHF and DM-uncomp flagged; CVD (post-pred) NOT flagged.
    row_100 = result.row(0, named=True)
    assert row_100["Congestive heart failure"] is True
    assert row_100["Diabetes without chronic complications"] is True
    assert row_100["Cerebrovascular disease"] is False
    assert row_100["any_charlson"] is True
    assert row_100["n_categories"] == 2

    # Patient 200: CHF event is post-prediction → no flags.
    row_200 = result.row(1, named=True)
    assert row_200["Congestive heart failure"] is False
    assert row_200["any_charlson"] is False
    assert row_200["n_categories"] == 0

    # Patient 300: I10 (HTN) is not in Charlson → no flags.
    row_300 = result.row(2, named=True)
    assert row_300["any_charlson"] is False
    assert row_300["n_categories"] == 0


def test_assign_patient_charlson_hierarchy_dedups_diabetes(tmp_path: Path) -> None:
    """Charlson convention: 'Diabetes with complications' absorbs 'Diabetes
    without complications' for any patient flagged for both."""
    from medrap.comorbidity import assign_patient_charlson, load_charlson_lookup

    cohort = tmp_path / "MEDS_cohort"
    _write_synthetic_meds_shard(
        cohort / "data" / "train" / "shard0.parquet",
        events=[
            (1, datetime(2020, 1, 1), "DIAGNOSIS//ICD//10//E110"),  # uncomp
            (1, datetime(2020, 1, 2), "DIAGNOSIS//ICD//10//E114"),  # complicated
        ],
    )
    val_schema = pl.DataFrame({"subject_id": [1], "prediction_time": [datetime(2021, 1, 1)]})
    result = assign_patient_charlson(cohort, val_schema, lookup=load_charlson_lookup())
    row = result.row(0, named=True)
    assert row["Diabetes with chronic complications"] is True
    # Hierarchy: complicated form absorbs uncomplicated.
    assert row["Diabetes without chronic complications"] is False
    assert row["n_categories"] == 1


def test_assign_patient_charlson_hierarchy_dedups_cancer(tmp_path: Path) -> None:
    """Metastatic solid tumor absorbs the 'Any malignancy' flag."""
    from medrap.comorbidity import assign_patient_charlson, load_charlson_lookup

    cohort = tmp_path / "MEDS_cohort"
    _write_synthetic_meds_shard(
        cohort / "data" / "train" / "shard0.parquet",
        events=[
            (1, datetime(2020, 1, 1), "DIAGNOSIS//ICD//10//C50"),  # breast cancer
            (1, datetime(2020, 1, 2), "DIAGNOSIS//ICD//10//C780"),  # metastatic
        ],
    )
    val_schema = pl.DataFrame({"subject_id": [1], "prediction_time": [datetime(2021, 1, 1)]})
    result = assign_patient_charlson(cohort, val_schema, lookup=load_charlson_lookup())
    row = result.row(0, named=True)
    assert row["Metastatic solid tumor"] is True
    assert row["Any malignancy"] is False
    assert row["n_categories"] == 1


def test_assign_patient_charlson_hierarchy_dedups_liver(tmp_path: Path) -> None:
    """Moderate/severe liver disease absorbs Mild liver disease."""
    from medrap.comorbidity import assign_patient_charlson, load_charlson_lookup

    cohort = tmp_path / "MEDS_cohort"
    _write_synthetic_meds_shard(
        cohort / "data" / "train" / "shard0.parquet",
        events=[
            (1, datetime(2020, 1, 1), "DIAGNOSIS//ICD//10//K74"),  # cirrhosis (mild)
            (1, datetime(2020, 1, 2), "DIAGNOSIS//ICD//10//K721"),  # chronic hepatic failure (mod-severe)
        ],
    )
    val_schema = pl.DataFrame({"subject_id": [1], "prediction_time": [datetime(2021, 1, 1)]})
    result = assign_patient_charlson(cohort, val_schema, lookup=load_charlson_lookup())
    row = result.row(0, named=True)
    assert row["Moderate or severe liver disease"] is True
    assert row["Mild liver disease"] is False
    assert row["n_categories"] == 1


def test_assign_patient_charlson_handles_icd9_codes(tmp_path: Path) -> None:
    """ICD-9 prefix matching works (decimals stripped)."""
    from medrap.comorbidity import assign_patient_charlson, load_charlson_lookup

    cohort = tmp_path / "MEDS_cohort"
    _write_synthetic_meds_shard(
        cohort / "data" / "train" / "shard0.parquet",
        events=[
            # ICD-9 428.0 (CHF) and 410.0 (MI), both pre-prediction
            (7, datetime(2010, 5, 1), "DIAGNOSIS//ICD//9//4280"),
            (7, datetime(2010, 5, 2), "DIAGNOSIS//ICD//9//4100"),
        ],
    )
    val_schema = pl.DataFrame({"subject_id": [7], "prediction_time": [datetime(2010, 6, 1)]})
    result = assign_patient_charlson(cohort, val_schema, lookup=load_charlson_lookup())
    row = result.row(0, named=True)
    assert row["Congestive heart failure"] is True
    assert row["Myocardial infarction"] is True
    assert row["n_categories"] == 2


def test_assign_patient_charlson_preserves_val_schema_row_order(tmp_path: Path) -> None:
    """Output row order must match val_schema row order so the result aligns 1:1 with the extraction
    artifacts."""
    from medrap.comorbidity import assign_patient_charlson, load_charlson_lookup

    cohort = tmp_path / "MEDS_cohort"
    _write_synthetic_meds_shard(
        cohort / "data" / "train" / "shard0.parquet",
        events=[
            (10, datetime(2020, 1, 1), "DIAGNOSIS//ICD//10//I50"),
            (20, datetime(2020, 1, 1), "DIAGNOSIS//ICD//10//N18"),
        ],
    )
    val_schema = pl.DataFrame(
        {
            "subject_id": [20, 10, 20],
            "prediction_time": [
                datetime(2021, 1, 1),
                datetime(2021, 1, 1),
                datetime(2021, 1, 1),
            ],
        }
    )
    result = assign_patient_charlson(cohort, val_schema, lookup=load_charlson_lookup())
    assert result["subject_id"].to_list() == [20, 10, 20]
    assert result.row(0, named=True)["Renal disease"] is True
    assert result.row(1, named=True)["Congestive heart failure"] is True
    assert result.row(2, named=True)["Renal disease"] is True
