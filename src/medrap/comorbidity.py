"""Charlson Comorbidity Index flags for MEDS patient cohorts.

Provides the 17 Charlson categories (Charlson 1987, Quan 2005 ICD-9-CM/
ICD-10 mapping) and a polars-backed function to flag every patient in a
validation schema based on their pre-prediction-time MEDS
``DIAGNOSIS//ICD//<version>//<code>`` events. Multi-membership is allowed
(a single patient can be flagged for several categories); the Charlson
hierarchy is applied so that "with complications" subsumes "without
complications" and the metastatic / moderate-severe variants subsume their
milder sibling categories per the canonical Charlson convention.

Sources:
    Charlson ME, et al. "A new method of classifying prognostic comorbidity
    in longitudinal studies: development and validation."
    J Chronic Dis. 1987;40(5):373-383.

    Quan H, et al. "Coding Algorithms for Defining Comorbidities in
    ICD-9-CM and ICD-10 Administrative Data." Medical Care.
    2005;43(11):1130-1139.

The mapping lives inline as Python dicts so it can be reviewed and amended
in source control. ``load_charlson_lookup`` optionally accepts a CSV path
(columns ``category, icd_version, icd_prefix``) for swapping in a
different mapping without code changes.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


# ---------------------------------------------------------------------------
# Canonical 17 Charlson categories in standard display order.
# ---------------------------------------------------------------------------

CHARLSON_CATEGORIES: tuple[str, ...] = (
    "Myocardial infarction",
    "Congestive heart failure",
    "Peripheral vascular disease",
    "Cerebrovascular disease",
    "Dementia",
    "Chronic pulmonary disease",
    "Rheumatologic disease",
    "Peptic ulcer disease",
    "Mild liver disease",
    "Diabetes without chronic complications",
    "Diabetes with chronic complications",
    "Hemiplegia or paraplegia",
    "Renal disease",
    "Any malignancy",
    "Moderate or severe liver disease",
    "Metastatic solid tumor",
    "AIDS/HIV",
)


# Hierarchy rules per Charlson convention: when a patient matches both
# the "lesser" and "greater" category, only the greater is counted.
_CHARLSON_HIERARCHY: tuple[tuple[str, str], ...] = (
    ("Diabetes with chronic complications", "Diabetes without chronic complications"),
    ("Moderate or severe liver disease", "Mild liver disease"),
    ("Metastatic solid tumor", "Any malignancy"),
)


# ---------------------------------------------------------------------------
# ICD-10-CM prefix → category (Quan 2005, Charlson section). Prefix-startswith
# match against the raw code part of ``DIAGNOSIS//ICD//10//<code>``.
# ---------------------------------------------------------------------------

_ICD10_PREFIXES: dict[str, list[str]] = {
    "Myocardial infarction": ["I21", "I22", "I252"],
    "Congestive heart failure": [
        "I099",
        "I110",
        "I130",
        "I132",
        "I255",
        "I420",
        "I425",
        "I426",
        "I427",
        "I428",
        "I429",
        "I43",
        "I50",
        "P290",
    ],
    "Peripheral vascular disease": [
        "I70",
        "I71",
        "I731",
        "I738",
        "I739",
        "I771",
        "I790",
        "I792",
        "K551",
        "K558",
        "K559",
        "Z958",
        "Z959",
    ],
    "Cerebrovascular disease": [
        "G45",
        "G46",
        "H340",
        "I60",
        "I61",
        "I62",
        "I63",
        "I64",
        "I65",
        "I66",
        "I67",
        "I68",
        "I69",
    ],
    "Dementia": ["F00", "F01", "F02", "F03", "F051", "G30", "G311"],
    "Chronic pulmonary disease": [
        "I278",
        "I279",
        "J40",
        "J41",
        "J42",
        "J43",
        "J44",
        "J45",
        "J46",
        "J47",
        "J60",
        "J61",
        "J62",
        "J63",
        "J64",
        "J65",
        "J66",
        "J67",
        "J684",
        "J701",
        "J703",
    ],
    "Rheumatologic disease": [
        "M05",
        "M06",
        "M315",
        "M32",
        "M33",
        "M34",
        "M351",
        "M353",
        "M360",
    ],
    "Peptic ulcer disease": ["K25", "K26", "K27", "K28"],
    "Mild liver disease": [
        "B18",
        "K700",
        "K701",
        "K702",
        "K703",
        "K709",
        "K717",
        "K73",
        "K74",
        "K760",
        "K762",
        "K763",
        "K764",
        "K768",
        "K769",
        "Z944",
    ],
    "Diabetes without chronic complications": [
        "E100",
        "E101",
        "E106",
        "E108",
        "E109",
        "E110",
        "E111",
        "E116",
        "E118",
        "E119",
        "E120",
        "E121",
        "E126",
        "E128",
        "E129",
        "E130",
        "E131",
        "E136",
        "E138",
        "E139",
        "E140",
        "E141",
        "E146",
        "E148",
        "E149",
    ],
    "Diabetes with chronic complications": [
        "E102",
        "E103",
        "E104",
        "E105",
        "E107",
        "E112",
        "E113",
        "E114",
        "E115",
        "E117",
        "E122",
        "E123",
        "E124",
        "E125",
        "E127",
        "E132",
        "E133",
        "E134",
        "E135",
        "E137",
        "E142",
        "E143",
        "E144",
        "E145",
        "E147",
    ],
    "Hemiplegia or paraplegia": [
        "G041",
        "G114",
        "G801",
        "G802",
        "G81",
        "G82",
        "G830",
        "G831",
        "G832",
        "G833",
        "G834",
        "G839",
    ],
    "Renal disease": [
        "I120",
        "I131",
        "N18",
        "N19",
        "N250",
        "Z490",
        "Z491",
        "Z492",
        "Z940",
        "Z992",
    ],
    "Any malignancy": [
        "C00",
        "C01",
        "C02",
        "C03",
        "C04",
        "C05",
        "C06",
        "C07",
        "C08",
        "C09",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "C25",
        "C26",
        "C30",
        "C31",
        "C32",
        "C33",
        "C34",
        "C37",
        "C38",
        "C39",
        "C40",
        "C41",
        "C43",
        "C45",
        "C46",
        "C47",
        "C48",
        "C49",
        "C50",
        "C51",
        "C52",
        "C53",
        "C54",
        "C55",
        "C56",
        "C57",
        "C58",
        "C60",
        "C61",
        "C62",
        "C63",
        "C64",
        "C65",
        "C66",
        "C67",
        "C68",
        "C69",
        "C70",
        "C71",
        "C72",
        "C73",
        "C74",
        "C75",
        "C76",
        "C81",
        "C82",
        "C83",
        "C84",
        "C85",
        "C88",
        "C90",
        "C91",
        "C92",
        "C93",
        "C94",
        "C95",
        "C96",
        "C97",
    ],
    "Moderate or severe liver disease": [
        "I85",
        "I864",
        "K704",
        "K711",
        "K721",
        "K729",
        "K765",
        "K766",
        "K767",
    ],
    "Metastatic solid tumor": ["C77", "C78", "C79", "C80"],
    "AIDS/HIV": ["B20", "B21", "B22", "B24"],
}


# ---------------------------------------------------------------------------
# ICD-9-CM prefix → category (Quan 2005, Charlson section). MIMIC strips
# decimals, so ``428.0`` → ``4280``; all prefixes follow that convention.
# ---------------------------------------------------------------------------

_ICD9_PREFIXES: dict[str, list[str]] = {
    "Myocardial infarction": ["410", "412"],
    "Congestive heart failure": [
        "39891",
        "40201",
        "40211",
        "40291",
        "40401",
        "40403",
        "40411",
        "40413",
        "40491",
        "40493",
        "4254",
        "4255",
        "4256",
        "4257",
        "4258",
        "4259",
        "428",
    ],
    "Peripheral vascular disease": [
        "0930",
        "4373",
        "440",
        "441",
        "4431",
        "4432",
        "4438",
        "4439",
        "4471",
        "5571",
        "5579",
        "V434",
    ],
    "Cerebrovascular disease": [
        "36234",
        "430",
        "431",
        "432",
        "433",
        "434",
        "435",
        "436",
        "437",
        "438",
    ],
    "Dementia": ["290", "2941", "3312"],
    "Chronic pulmonary disease": [
        "4168",
        "4169",
        "490",
        "491",
        "492",
        "493",
        "494",
        "495",
        "496",
        "500",
        "501",
        "502",
        "503",
        "504",
        "505",
        "5064",
        "5081",
        "5088",
    ],
    "Rheumatologic disease": [
        "4465",
        "7100",
        "7101",
        "7102",
        "7103",
        "7104",
        "7140",
        "7141",
        "7142",
        "7148",
        "7252",
    ],
    "Peptic ulcer disease": ["531", "532", "533", "534"],
    "Mild liver disease": [
        "07022",
        "07023",
        "07032",
        "07033",
        "07044",
        "07054",
        "0706",
        "0709",
        "570",
        "571",
        "5733",
        "5734",
        "5738",
        "5739",
        "V427",
    ],
    "Diabetes without chronic complications": [
        "2500",
        "2501",
        "2502",
        "2503",
        "2508",
        "2509",
    ],
    "Diabetes with chronic complications": [
        "2504",
        "2505",
        "2506",
        "2507",
    ],
    "Hemiplegia or paraplegia": [
        "3341",
        "342",
        "343",
        "3440",
        "3441",
        "3442",
        "3443",
        "3444",
        "3445",
        "3446",
        "3449",
    ],
    "Renal disease": [
        "40301",
        "40311",
        "40391",
        "40402",
        "40403",
        "40412",
        "40413",
        "40492",
        "40493",
        "582",
        "583",
        "585",
        "586",
        "5880",
        "V420",
        "V451",
        "V56",
    ],
    "Any malignancy": [
        "140",
        "141",
        "142",
        "143",
        "144",
        "145",
        "146",
        "147",
        "148",
        "149",
        "150",
        "151",
        "152",
        "153",
        "154",
        "155",
        "156",
        "157",
        "158",
        "159",
        "160",
        "161",
        "162",
        "163",
        "164",
        "165",
        "166",
        "167",
        "168",
        "169",
        "170",
        "171",
        "172",
        "174",
        "175",
        "176",
        "177",
        "178",
        "179",
        "180",
        "181",
        "182",
        "183",
        "184",
        "185",
        "186",
        "187",
        "188",
        "189",
        "190",
        "191",
        "192",
        "193",
        "194",
        "195",
        "200",
        "201",
        "202",
        "203",
        "204",
        "205",
        "206",
        "207",
        "208",
        "2386",
    ],
    "Moderate or severe liver disease": [
        "4560",
        "4561",
        "4562",
        "5722",
        "5723",
        "5724",
        "5728",
    ],
    "Metastatic solid tumor": ["196", "197", "198", "199"],
    "AIDS/HIV": ["042", "043", "044"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_charlson_lookup(
    csv_path: Path | None = None,
) -> dict[tuple[int, str], frozenset[str]]:
    """Return ``{(icd_version, icd_prefix): frozenset[category]}``.

    By default uses the inline Quan-2005 Charlson mapping. Pass ``csv_path``
    to load a different mapping (CSV with columns
    ``category, icd_version, icd_prefix``).
    """
    if csv_path is None:
        accum: dict[tuple[int, str], set[str]] = defaultdict(set)
        for cat, prefixes in _ICD10_PREFIXES.items():
            for p in prefixes:
                accum[(10, p)].add(cat)
        for cat, prefixes in _ICD9_PREFIXES.items():
            for p in prefixes:
                accum[(9, p)].add(cat)
        return {k: frozenset(v) for k, v in accum.items()}

    accum2: dict[tuple[int, str], set[str]] = defaultdict(set)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v = int(row["icd_version"])
            except (KeyError, ValueError):
                continue
            cat = (row.get("category") or "").strip()
            prefix = (row.get("icd_prefix") or "").strip()
            if not cat or not prefix:
                continue
            accum2[(v, prefix)].add(cat)
    return {k: frozenset(v) for k, v in accum2.items()}


def lookup_categories(
    lookup: dict[tuple[int, str], frozenset[str]],
    icd_version: int,
    icd_code: str,
) -> frozenset[str]:
    """Return all Charlson categories matching ``icd_code`` by prefix.

    Multi-membership: a code may match multiple prefixes; the union of
    those prefix sets is returned. The hierarchy de-duplication (e.g.,
    diabetes-with-complications absorbs diabetes-without-complications) is
    applied later, only after all events for a patient have been aggregated
    — see :func:`assign_patient_charlson`.
    """
    if icd_code is None:
        return frozenset()
    matches: set[str] = set()
    for (v, prefix), cats in lookup.items():
        if v == icd_version and icd_code.startswith(prefix):
            matches.update(cats)
    return frozenset(matches)


def _apply_charlson_hierarchy(flags: set[str]) -> set[str]:
    """Apply the canonical Charlson "greater absorbs lesser" rules.

    Modifies and returns ``flags``. If a patient is flagged for both the
    lesser and greater category in a hierarchy pair, only the greater
    remains.
    """
    for greater, lesser in _CHARLSON_HIERARCHY:
        if greater in flags:
            flags.discard(lesser)
    return flags


def assign_patient_charlson(
    meds_cohort_dir: Path,
    val_schema: pl.DataFrame,
    *,
    lookup: dict[tuple[int, str], frozenset[str]] | None = None,
) -> pl.DataFrame:
    """Flag every (subject_id, prediction_time) row with Charlson comorbidities.

    Scans the MEDS cohort's ``data/*/*.parquet`` shards for
    ``DIAGNOSIS//ICD//<version>//<code>`` events belonging to subjects in
    ``val_schema``. For each ``val_schema`` row, accumulates the set of
    Charlson categories from events whose ``time <= prediction_time``, then
    applies the standard Charlson hierarchy de-duplication
    (e.g. metastatic absorbs any-malignancy).

    Args:
        meds_cohort_dir: Root of the MEDS cohort (containing ``data/``).
        val_schema: DataFrame with at least ``subject_id`` and
            ``prediction_time`` columns. Rows in the result preserve
            ``val_schema`` row order.
        lookup: Override the default Quan-2005 Charlson lookup
            (see :func:`load_charlson_lookup`).

    Returns:
        polars DataFrame with columns:
            ``subject_id``, ``prediction_time``,
            one bool per category in :data:`CHARLSON_CATEGORIES`,
            ``any_charlson`` (bool), ``n_categories`` (int).
        Row order matches ``val_schema`` row order 1:1.
    """
    import polars as pl  # local import keeps the module light

    if lookup is None:
        lookup = load_charlson_lookup()

    val_subjects = list({int(s) for s in val_schema["subject_id"].to_list()})

    glob = str(Path(meds_cohort_dir) / "data" / "*" / "*.parquet")
    events = (
        pl.scan_parquet(glob)
        .filter(pl.col("code").str.starts_with("DIAGNOSIS//ICD//") & pl.col("subject_id").is_in(val_subjects))
        .select(["subject_id", "time", "code"])
        .with_columns(
            pl.col("code")
            .str.extract(r"^DIAGNOSIS//ICD//(\d+)//", 1)
            .cast(pl.Int8, strict=False)
            .alias("icd_version"),
            pl.col("code").str.extract(r"^DIAGNOSIS//ICD//\d+//(.+)$", 1).alias("icd_code"),
        )
        .collect()
    )

    # Group prefixes by version so per-event matching is O(prefixes_per_version).
    prefixes_by_version: dict[int, list[tuple[str, frozenset[str]]]] = defaultdict(list)
    for (version, prefix), cats in lookup.items():
        prefixes_by_version[int(version)].append((prefix, cats))

    # Per-event compute set of matching categories.
    subject_events: dict[int, list[tuple[object, frozenset[str]]]] = defaultdict(list)
    for row in events.iter_rows(named=True):
        v = row.get("icd_version")
        code = row.get("icd_code")
        if v is None or code is None:
            continue
        matches: set[str] = set()
        for prefix, cats in prefixes_by_version.get(int(v), []):
            if code.startswith(prefix):
                matches.update(cats)
        if matches:
            subject_events[int(row["subject_id"])].append((row["time"], frozenset(matches)))

    out_rows: list[dict] = []
    for vrow in val_schema.iter_rows(named=True):
        sid = int(vrow["subject_id"])
        pred_t = vrow["prediction_time"]
        flags: set[str] = set()
        for ev_t, ev_cats in subject_events.get(sid, []):
            if ev_t is None or pred_t is None:
                continue
            if ev_t <= pred_t:
                flags |= ev_cats
        # Apply the Charlson hierarchy (greater absorbs lesser).
        _apply_charlson_hierarchy(flags)
        row: dict = {"subject_id": sid, "prediction_time": pred_t}
        for cat in CHARLSON_CATEGORIES:
            row[cat] = cat in flags
        row["any_charlson"] = bool(flags)
        row["n_categories"] = len(flags)
        out_rows.append(row)

    return pl.DataFrame(
        out_rows,
        schema={
            "subject_id": pl.Int64,
            "prediction_time": val_schema.schema["prediction_time"],
            **dict.fromkeys(CHARLSON_CATEGORIES, pl.Boolean),
            "any_charlson": pl.Boolean,
            "n_categories": pl.Int64,
        },
    )
