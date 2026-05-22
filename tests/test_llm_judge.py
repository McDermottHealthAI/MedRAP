"""Tests for the LLM-as-a-judge retrieval-relevance evaluation module.

All tests use the in-memory :class:`FakeJudge` + tiny polars / HF Dataset
fixtures. No live OpenAI call is made. Structure mirrors the 31-item
spec in ``D3_plan.md`` § "Test layout".
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from datasets import Dataset

from medrap.llm_judge import (
    FakeJudge,
    Judge,
    JudgePair,
    JudgePromptBuilder,
    OpenAIJudge,
    PatientTimelineRenderer,
    Verdict,
    _clean_lab_description,
    _render_patient_narrative,
    build_human_validation_subset,
    build_pairs,
    build_per_patient_rollup,
    compute_patient_clinical_summary,
    run_judge,
    summarize_winrates,
    write_results_workbook,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_artifacts(
    *,
    n_patients: int,
    k: int,
    labels: list[int] | None = None,
    top1_doc_ids: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Return a minimally-populated artifacts dict with deterministic ids."""
    if labels is None:
        labels = [i % 2 for i in range(n_patients)]
    if top1_doc_ids is None:
        top1_doc_ids = [100 + i for i in range(n_patients)]

    # doc_ids shape (N, R=1, K); doc_scores sorted descending per row.
    doc_ids = np.zeros((n_patients, 1, k), dtype=np.int64)
    doc_scores = np.zeros((n_patients, 1, k), dtype=np.float32)
    for i in range(n_patients):
        doc_ids[i, 0, 0] = top1_doc_ids[i]
        for j in range(1, k):
            doc_ids[i, 0, j] = top1_doc_ids[i] + 1000 * j
        doc_scores[i, 0, :] = np.linspace(1.0, 0.1, k, dtype=np.float32)

    return {
        "doc_ids": doc_ids,
        "doc_scores": doc_scores,
        "targets": np.array(labels, dtype=np.float32),
        "logits": np.random.RandomState(0).randn(n_patients, 2).astype(np.float32),
        "query_embeddings": np.zeros((n_patients, 1, 4), dtype=np.float32),
    }


def _make_val_schema(subject_ids: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": subject_ids,
            "end_event_index": list(range(len(subject_ids))),
            "prediction_time": [datetime(2024, 1, 1) for _ in subject_ids],
        }
    )


def _save_retrieval_ds(tmp_path: Path, titles: list[str], extra_cols: dict | None = None) -> Path:
    data: dict = {"title": titles, "doc_text": [f"text for {t}" for t in titles]}
    data["doc_ids"] = [100 + i for i in range(len(titles))]
    if extra_cols:
        data.update(extra_cols)
    dataset_path = tmp_path / "retrieval_db"
    Dataset.from_dict(data).save_to_disk(str(dataset_path))
    return dataset_path


def _save_codes_parquet(tmp_path: Path, codes: list[str], descriptions: list[str | None]) -> Path:
    path = tmp_path / "codes.parquet"
    pl.DataFrame({"code": codes, "description": descriptions}).write_parquet(path)
    return path


def _make_meds_cohort(
    tmp_path: Path,
    subject_events: dict[int, list[tuple]],
) -> Path:
    """Write a one-shard cohort with the given events per subject.

    Each tuple may be ``(time, code)`` or ``(time, code, numeric_value)``;
    a null ``numeric_value`` is written when omitted.
    """
    cohort = tmp_path / "meds_cohort"
    data_dir = cohort / "data" / "train"
    data_dir.mkdir(parents=True)
    rows = []
    for sid, events in subject_events.items():
        for ev in events:
            if len(ev) == 2:
                t, code = ev
                val = None
            else:
                t, code, val = ev
            rows.append(
                {"subject_id": sid, "time": t, "code": code, "numeric_value": val}
            )
    pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime,
            "code": pl.Utf8,
            "numeric_value": pl.Float64,
        },
    ).write_parquet(data_dir / "shard_0.parquet")
    return cohort


# ---------------------------------------------------------------------------
# T1–T8: build_pairs
# ---------------------------------------------------------------------------


def test_build_pairs_f1_uses_top1_and_random_other() -> None:
    artifacts = _make_artifacts(n_patients=20, k=5, labels=[0] * 10 + [1] * 10)
    schema = _make_val_schema(list(range(20)))
    pairs = build_pairs(
        artifacts=artifacts,
        val_schema=schema,
        labels=artifacts["targets"].astype(int),
        families=("F1",),
        n_patients=10,
        pairs_per_patient_per_family=1,
        corpus_size=10_000,
        k=5,
        seed=0,
    )
    assert len(pairs) == 10
    for p in pairs:
        assert p.family == "F1"
        assert p.target_doc_id == artifacts["doc_ids"][p.anchor_row_idx, 0, 0]
        assert p.other_doc_id != p.target_doc_id


def test_build_pairs_f2_samples_lower_rank_same_patient() -> None:
    artifacts = _make_artifacts(n_patients=20, k=5, labels=[0] * 10 + [1] * 10)
    schema = _make_val_schema(list(range(20)))
    pairs = build_pairs(
        artifacts=artifacts,
        val_schema=schema,
        labels=artifacts["targets"].astype(int),
        families=("F2",),
        n_patients=10,
        pairs_per_patient_per_family=1,
        corpus_size=10_000,
        k=5,
        seed=0,
    )
    for p in pairs:
        assert p.family == "F2"
        row = p.anchor_row_idx
        top1 = int(artifacts["doc_ids"][row, 0, 0])
        lower_ranks = {int(artifacts["doc_ids"][row, 0, j]) for j in range(1, 5)}
        assert p.target_doc_id == top1
        assert p.other_doc_id in lower_ranks
        assert p.other_rank is not None and 2 <= p.other_rank <= 5


def test_build_pairs_f2_skipped_silently_when_k_equals_one() -> None:
    artifacts = _make_artifacts(n_patients=10, k=1, labels=[0] * 5 + [1] * 5)
    schema = _make_val_schema(list(range(10)))
    pairs = build_pairs(
        artifacts=artifacts,
        val_schema=schema,
        labels=artifacts["targets"].astype(int),
        families=("F1", "F2"),
        n_patients=6,
        pairs_per_patient_per_family=1,
        corpus_size=1_000,
        k=1,
        seed=0,
    )
    families = {p.family for p in pairs}
    assert "F2" not in families
    assert "F1" in families


def test_build_pairs_f3_other_patient_has_same_label() -> None:
    artifacts = _make_artifacts(n_patients=20, k=3, labels=[0] * 10 + [1] * 10)
    schema = _make_val_schema(list(range(20)))
    pairs = build_pairs(
        artifacts=artifacts,
        val_schema=schema,
        labels=artifacts["targets"].astype(int),
        families=("F3",),
        n_patients=10,
        pairs_per_patient_per_family=1,
        corpus_size=10_000,
        k=3,
        seed=0,
    )
    labels = artifacts["targets"].astype(int)
    for p in pairs:
        assert p.other_source_row_idx is not None
        assert labels[p.other_source_row_idx] == p.anchor_label
        assert p.other_source_row_idx != p.anchor_row_idx


def test_build_pairs_f4_other_patient_has_opposite_label() -> None:
    artifacts = _make_artifacts(n_patients=20, k=3, labels=[0] * 10 + [1] * 10)
    schema = _make_val_schema(list(range(20)))
    pairs = build_pairs(
        artifacts=artifacts,
        val_schema=schema,
        labels=artifacts["targets"].astype(int),
        families=("F4",),
        n_patients=10,
        pairs_per_patient_per_family=1,
        corpus_size=10_000,
        k=3,
        seed=0,
    )
    labels = artifacts["targets"].astype(int)
    for p in pairs:
        assert p.other_source_row_idx is not None
        assert labels[p.other_source_row_idx] == 1 - p.anchor_label


def test_build_pairs_deduplicates_identical_target_and_other_doc() -> None:
    # Force every anchor's top-1 to be the same doc so F3 dedupe kicks in hard.
    artifacts = _make_artifacts(
        n_patients=20,
        k=3,
        labels=[0] * 10 + [1] * 10,
        top1_doc_ids=[42] * 20,
    )
    schema = _make_val_schema(list(range(20)))
    pairs = build_pairs(
        artifacts=artifacts,
        val_schema=schema,
        labels=artifacts["targets"].astype(int),
        families=("F3",),
        n_patients=10,
        pairs_per_patient_per_family=1,
        corpus_size=10_000,
        k=3,
        seed=0,
        dedupe_identical_docs=True,
    )
    # With a shared top-1, same-label redraw can't escape → most/all skip.
    for p in pairs:
        assert p.target_doc_id != p.other_doc_id


def test_build_pairs_is_deterministic_with_seed() -> None:
    artifacts = _make_artifacts(n_patients=20, k=5, labels=[0] * 10 + [1] * 10)
    schema = _make_val_schema(list(range(20)))
    kwargs = dict(
        artifacts=artifacts,
        val_schema=schema,
        labels=artifacts["targets"].astype(int),
        families=("F1", "F2", "F3", "F4"),
        n_patients=8,
        pairs_per_patient_per_family=1,
        corpus_size=10_000,
        k=5,
        seed=123,
    )
    a = build_pairs(**kwargs)
    b = build_pairs(**kwargs)
    assert [p.pair_id for p in a] == [p.pair_id for p in b]
    assert [(p.target_doc_id, p.other_doc_id, p.target_position) for p in a] == [
        (p.target_doc_id, p.other_doc_id, p.target_position) for p in b
    ]


def test_build_pairs_ab_position_roughly_balanced() -> None:
    artifacts = _make_artifacts(n_patients=200, k=5, labels=[0] * 100 + [1] * 100)
    schema = _make_val_schema(list(range(200)))
    pairs = build_pairs(
        artifacts=artifacts,
        val_schema=schema,
        labels=artifacts["targets"].astype(int),
        families=("F1",),
        n_patients=200,
        pairs_per_patient_per_family=1,
        corpus_size=10_000,
        k=5,
        seed=7,
    )
    frac_a = sum(1 for p in pairs if p.target_position == "A") / len(pairs)
    assert 0.4 <= frac_a <= 0.6


def test_stratified_patient_sample_respects_label_balance() -> None:
    artifacts = _make_artifacts(n_patients=40, k=3, labels=[0] * 20 + [1] * 20)
    schema = _make_val_schema(list(range(40)))
    pairs = build_pairs(
        artifacts=artifacts,
        val_schema=schema,
        labels=artifacts["targets"].astype(int),
        families=("F1",),
        n_patients=20,
        pairs_per_patient_per_family=1,
        corpus_size=10_000,
        k=3,
        seed=0,
    )
    anchor_labels = [p.anchor_label for p in pairs]
    assert sum(anchor_labels) == 10  # 50/50 split


# ---------------------------------------------------------------------------
# T10–T11: run_judge
# ---------------------------------------------------------------------------


def _mk_verdict(pair_id: str, winner: str, confidence: float = 1.0) -> Verdict:
    return Verdict(
        pair_id=pair_id,
        winner_position=winner,  # type: ignore[arg-type]
        target_won=None,  # run_judge sets this
        confidence=confidence,
        rationale="",
        raw_response="{}",
        model="fake",
        prompt_tokens=0,
        completion_tokens=0,
    )


def test_run_judge_roundtrips_fake_verdicts_to_dataframe(tmp_path: Path) -> None:
    artifacts = _make_artifacts(n_patients=4, k=3, labels=[0, 0, 1, 1])
    schema = _make_val_schema([1, 2, 3, 4])
    pairs = build_pairs(
        artifacts=artifacts,
        val_schema=schema,
        labels=artifacts["targets"].astype(int),
        families=("F1",),
        n_patients=4,
        pairs_per_patient_per_family=1,
        corpus_size=1000,
        k=3,
        seed=0,
    )

    codes_fp = _save_codes_parquet(tmp_path, ["A", "B"], ["desc A", "desc B"])
    retrieval_ds_path = _save_retrieval_ds(tmp_path, titles=[f"t{i}" for i in range(2000)])
    from datasets import load_from_disk

    ds = load_from_disk(str(retrieval_ds_path))
    builder = JudgePromptBuilder(
        task_description="test task",
        timeline_renderer=PatientTimelineRenderer(codes_parquet=codes_fp),
        retrieval_ds=ds,
        doc_id_to_row={d: i for i, d in enumerate(ds["doc_ids"])},
    )
    judge = FakeJudge.always_A()
    df = run_judge(pairs, judge=judge, prompt_builder=builder, max_workers=1, progress=False)
    assert isinstance(df, pl.DataFrame)
    assert set(["pair_id", "family", "target_won", "winner_position"]).issubset(df.columns)
    assert df.height == len(pairs)


def test_run_judge_marks_target_won_based_on_target_position(tmp_path: Path) -> None:
    artifacts = _make_artifacts(n_patients=4, k=3)
    schema = _make_val_schema([10, 11, 12, 13])
    pairs = build_pairs(
        artifacts=artifacts,
        val_schema=schema,
        labels=artifacts["targets"].astype(int),
        families=("F1",),
        n_patients=4,
        pairs_per_patient_per_family=1,
        corpus_size=1000,
        k=3,
        seed=0,
    )
    codes_fp = _save_codes_parquet(tmp_path, ["A"], ["desc"])
    retrieval_ds_path = _save_retrieval_ds(tmp_path, titles=[f"t{i}" for i in range(2000)])
    from datasets import load_from_disk

    ds = load_from_disk(str(retrieval_ds_path))
    builder = JudgePromptBuilder(
        task_description="test task",
        timeline_renderer=PatientTimelineRenderer(codes_parquet=codes_fp),
        retrieval_ds=ds,
        doc_id_to_row={d: i for i, d in enumerate(ds["doc_ids"])},
    )
    judge = FakeJudge.always_A()
    df = run_judge(pairs, judge=judge, prompt_builder=builder, max_workers=1, progress=False)
    df_pairs = {p.pair_id: p for p in pairs}
    for row in df.iter_rows(named=True):
        expected = df_pairs[row["pair_id"]].target_position == "A"
        assert row["target_won"] == expected


# ---------------------------------------------------------------------------
# T12–T13: prompt builder
# ---------------------------------------------------------------------------


def test_prompt_builder_places_target_according_to_position(tmp_path: Path) -> None:
    codes_fp = _save_codes_parquet(tmp_path, ["A"], ["desc"])
    retrieval_ds_path = _save_retrieval_ds(tmp_path, titles=["doc-target", "doc-other"])
    from datasets import load_from_disk

    ds = load_from_disk(str(retrieval_ds_path))
    builder = JudgePromptBuilder(
        task_description="Predict outcome X.",
        timeline_renderer=PatientTimelineRenderer(codes_parquet=codes_fp),
        retrieval_ds=ds,
        doc_id_to_row={100: 0, 101: 1},
    )

    pair_a = JudgePair(
        pair_id="p1", family="F1", anchor_row_idx=0, anchor_subject_id=1, anchor_label=1,
        target_doc_id=100, other_doc_id=101, target_position="A",
    )
    pair_b = JudgePair(
        pair_id="p2", family="F1", anchor_row_idx=0, anchor_subject_id=1, anchor_label=1,
        target_doc_id=100, other_doc_id=101, target_position="B",
    )
    _, up_a = builder.build(pair_a, patient_timeline="EV1\nEV2")
    _, up_b = builder.build(pair_b, patient_timeline="EV1\nEV2")
    assert up_a.index("text for doc-target") < up_a.index("text for doc-other")
    assert up_b.index("text for doc-other") < up_b.index("text for doc-target")


def test_prompt_builder_uses_xml_delimited_sections_and_rubric(tmp_path: Path) -> None:
    codes_fp = _save_codes_parquet(tmp_path, ["A"], ["desc"])
    retrieval_ds_path = _save_retrieval_ds(tmp_path, titles=["doc-target", "doc-other"])
    from datasets import load_from_disk

    ds = load_from_disk(str(retrieval_ds_path))
    builder = JudgePromptBuilder(
        task_description="Predict outcome X.",
        timeline_renderer=PatientTimelineRenderer(codes_parquet=codes_fp),
        retrieval_ds=ds,
        doc_id_to_row={100: 0, 101: 1},
    )
    pair = JudgePair(
        pair_id="p1", family="F1", anchor_row_idx=0, anchor_subject_id=1, anchor_label=1,
        target_doc_id=100, other_doc_id=101, target_position="A",
    )
    sys_p, user_p = builder.build(pair, patient_timeline="PATIENT: 50yo M")
    # XML-delimited sections in the user prompt.
    for tag in ("<task>", "</task>", "<patient>", "</patient>",
                "<document_a>", "</document_a>", "<document_b>", "</document_b>"):
        assert tag in user_p, f"missing tag {tag}"
    # Section order: task → patient → doc A → doc B.
    assert (
        user_p.index("<task>")
        < user_p.index("<patient>")
        < user_p.index("<document_a>")
        < user_p.index("<document_b>")
    )
    # Rubric + bias controls live in the system prompt.
    assert "EVALUATION CRITERIA" in sys_p
    assert "BIAS CONTROLS" in sys_p
    assert "Patient specificity" in sys_p


def test_prompt_builder_truncates_doc_texts_to_max_chars(tmp_path: Path) -> None:
    codes_fp = _save_codes_parquet(tmp_path, ["A"], ["desc"])
    long_text = "x" * 20_000
    retrieval_ds_path = tmp_path / "retrieval_db"
    Dataset.from_dict({"title": ["t"], "doc_text": [long_text], "doc_ids": [42]}).save_to_disk(
        str(retrieval_ds_path)
    )
    from datasets import load_from_disk

    ds = load_from_disk(str(retrieval_ds_path))
    builder = JudgePromptBuilder(
        task_description="test",
        timeline_renderer=PatientTimelineRenderer(codes_parquet=codes_fp),
        retrieval_ds=ds,
        doc_id_to_row={42: 0},
        max_doc_chars=100,
    )
    pair = JudgePair(
        pair_id="p1", family="F1", anchor_row_idx=0, anchor_subject_id=1, anchor_label=1,
        target_doc_id=42, other_doc_id=42, target_position="A",
    )
    _, user_prompt = builder.build(pair, patient_timeline="")
    assert "x" * 101 not in user_prompt
    assert "x" * 100 in user_prompt


# ---------------------------------------------------------------------------
# T14–T15a: timeline renderer
# ---------------------------------------------------------------------------


def test_timeline_renderer_returns_last_n_events_before_prediction_time(tmp_path: Path) -> None:
    codes_fp = _save_codes_parquet(
        tmp_path,
        ["LAB//E1", "LAB//E2", "LAB//E3", "LAB//E4", "LAB//E5"],
        ["d1", "d2", "d3", "d4", "d5"],
    )
    cohort = _make_meds_cohort(
        tmp_path,
        {
            42: [
                (datetime(2024, 1, 1), "LAB//E1"),
                (datetime(2024, 1, 2), "LAB//E2"),
                (datetime(2024, 1, 3), "LAB//E3"),
                (datetime(2024, 1, 4), "LAB//E4"),
                (datetime(2024, 1, 5), "LAB//E5"),
            ]
        },
    )
    renderer = PatientTimelineRenderer(codes_parquet=codes_fp, max_events=3)
    text = renderer.render(42, datetime(2024, 1, 4), cohort)
    # Header + timeline header + 3 tail events (d2, d3, d4).
    rendered_lines = [line for line in text.splitlines() if line.startswith("Lab:")]
    assert rendered_lines == ["Lab: d2", "Lab: d3", "Lab: d4"]


def test_timeline_drops_null_description_events(tmp_path: Path) -> None:
    """Events whose code has a null description in codes.parquet are noise
    (raw MIMIC itemids like ``LAB//227944//UNK``) and must be dropped."""
    codes_fp = _save_codes_parquet(
        tmp_path,
        ["LAB//GOOD", "LAB//BAD//UNK", "DIAGNOSIS//ICD10//I509"],
        ["Hemoglobin", None, "Heart failure, unspecified"],
    )
    cohort = _make_meds_cohort(
        tmp_path,
        {
            7: [
                (datetime(2024, 1, 1), "LAB//BAD//UNK"),
                (datetime(2024, 1, 2), "LAB//GOOD"),
                (datetime(2024, 1, 3), "LAB//BAD//UNK"),
                (datetime(2024, 1, 4), "DIAGNOSIS//ICD10//I509"),
                (datetime(2024, 1, 5), "LAB//BAD//UNK"),
            ]
        },
    )
    renderer = PatientTimelineRenderer(codes_parquet=codes_fp, max_events=20)
    text = renderer.render(7, datetime(2025, 1, 1), cohort)
    assert "LAB//BAD" not in text
    assert "Hemoglobin" in text
    assert "Heart failure, unspecified" in text


def test_timeline_collapses_consecutive_duplicates(tmp_path: Path) -> None:
    """Six back-to-back lactate draws should collapse to one line."""
    codes_fp = _save_codes_parquet(
        tmp_path,
        ["LAB//LACTATE//mmol/L"],
        ["Lactate in Blood"],
    )
    events = [
        (datetime(2024, 1, 1, h, 0), "LAB//LACTATE//mmol/L", 2.0 + 0.1 * h)
        for h in range(6)
    ]
    cohort = _make_meds_cohort(tmp_path, {7: events})
    renderer = PatientTimelineRenderer(codes_parquet=codes_fp, max_events=20)
    text = renderer.render(7, datetime(2025, 1, 1), cohort)
    lactate_lines = [line for line in text.splitlines() if "Lactate" in line]
    assert len(lactate_lines) == 1
    # Dedup keeps the *latest* measurement (2.5).
    assert "2.50" in lactate_lines[0] or "2.5" in lactate_lines[0]
    assert "(mmol/L)" in lactate_lines[0]


def test_timeline_prepends_demographic_block(tmp_path: Path) -> None:
    """A demographic header should be rendered even when the static
    demographic codes themselves fell outside the last-N event window."""
    codes_fp = _save_codes_parquet(tmp_path, ["LAB//X"], ["Glucose"])
    cohort = _make_meds_cohort(
        tmp_path,
        {9: [(datetime(2024, 6, 1), "LAB//X", 105.0)]},
    )
    renderer = PatientTimelineRenderer(codes_parquet=codes_fp, max_events=20)
    demographics = {
        "subject_id": 9,
        "gender": "F",
        "race": "WHITE",
        "birth_time": datetime(1957, 6, 1),
    }
    text = renderer.render(9, datetime(2024, 6, 1), cohort, demographics=demographics)
    first_line = text.splitlines()[0]
    assert first_line.startswith("PATIENT: ")
    assert "67-year-old" in first_line
    assert "female" in first_line
    assert "WHITE" in first_line


def test_timeline_includes_numeric_values_with_units(tmp_path: Path) -> None:
    codes_fp = _save_codes_parquet(
        tmp_path,
        ["LAB//LACTATE//mmol/L", "LAB//PH//units", "PROCEDURE//START//CXR"],
        ["Lactate in Blood", "pH of Blood", "Plain chest X-ray"],
    )
    cohort = _make_meds_cohort(
        tmp_path,
        {
            1: [
                (datetime(2024, 1, 1, 8, 0), "PROCEDURE//START//CXR", None),
                (datetime(2024, 1, 1, 8, 30), "LAB//LACTATE//mmol/L", 5.4),
                (datetime(2024, 1, 1, 8, 31), "LAB//PH//units", 7.21),
            ]
        },
    )
    renderer = PatientTimelineRenderer(codes_parquet=codes_fp, max_events=20)
    text = renderer.render(1, datetime(2024, 1, 1, 12, 0), cohort)
    assert "Lab: Lactate in Blood = 5.40 (mmol/L)" in text
    assert "Lab: pH of Blood = 7.21" in text
    # PROCEDURE//START carries no numeric_value: no '=' suffix.
    assert "Procedure (start): Plain chest X-ray" in text
    assert "Procedure (start): Plain chest X-ray = " not in text


def test_timeline_collapses_multiline_descriptions(tmp_path: Path) -> None:
    """Some MIMIC codes (e.g. INFUSION_START itemids) have descriptions that
    span multiple preparations joined by '\\n'. The renderer must collapse
    them so later lines don't orphan without a type prefix."""
    codes_fp = _save_codes_parquet(
        tmp_path,
        ["INFUSION_START//225798"],
        ["vancomycin Injection\nvancomycin Oral Solution"],
    )
    cohort = _make_meds_cohort(
        tmp_path,
        {3: [(datetime(2024, 6, 1), "INFUSION_START//225798", None)]},
    )
    renderer = PatientTimelineRenderer(codes_parquet=codes_fp, max_events=20)
    text = renderer.render(3, datetime(2024, 6, 2), cohort)
    # The entire description belongs on a single labelled line.
    assert "Infusion start: vancomycin Injection / vancomycin Oral Solution" in text
    # No orphan continuation line without a type prefix.
    lines = [line for line in text.splitlines() if line.strip()]
    orphans = [line for line in lines if line.startswith("vancomycin")]
    assert orphans == []


def test_timeline_respects_max_events_after_filtering(tmp_path: Path) -> None:
    """``max_events`` caps the number of rendered events *after* filtering
    null-description events and deduplicating consecutive duplicates."""
    codes_fp = _save_codes_parquet(
        tmp_path,
        ["LAB//NOISE//UNK", "LAB//KEEP"] + [f"LAB//E{i}" for i in range(5)],
        [None, "kept"] + [f"d{i}" for i in range(5)],
    )
    events = [
        (datetime(2024, 1, 1, 0, 0), "LAB//NOISE//UNK"),
        (datetime(2024, 1, 2), "LAB//KEEP"),
        (datetime(2024, 1, 2, 0, 1), "LAB//KEEP"),  # consecutive dup → collapses
        (datetime(2024, 1, 3), "LAB//E0"),
        (datetime(2024, 1, 4), "LAB//E1"),
        (datetime(2024, 1, 5), "LAB//E2"),
        (datetime(2024, 1, 6), "LAB//E3"),
        (datetime(2024, 1, 7), "LAB//NOISE//UNK"),
    ]
    cohort = _make_meds_cohort(tmp_path, {5: events})
    renderer = PatientTimelineRenderer(codes_parquet=codes_fp, max_events=3)
    text = renderer.render(5, datetime(2025, 1, 1), cohort)
    rendered = [line for line in text.splitlines() if line.startswith("Lab:")]
    # After noise-drop + dedup we have [kept, d0, d1, d2, d3]; last 3 = [d1, d2, d3].
    assert rendered == ["Lab: d1", "Lab: d2", "Lab: d3"]


def test_codes_parquet_must_be_one_to_one(tmp_path: Path) -> None:
    codes_fp = _save_codes_parquet(tmp_path, ["DUP_CODE", "DUP_CODE", "OTHER"], ["d1", "d2", "d3"])
    with pytest.raises(ValueError, match="DUP_CODE"):
        PatientTimelineRenderer(codes_parquet=codes_fp)


# ---------------------------------------------------------------------------
# Clinical summary (healthcare utilization + chronic conditions)
# ---------------------------------------------------------------------------


def test_compute_patient_clinical_summary_counts_utilization_codes(tmp_path: Path) -> None:
    cohort = _make_meds_cohort(
        tmp_path,
        {
            42: [
                (datetime(2023, 1, 1), "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"),
                (datetime(2023, 1, 2), "ICU_ADMISSION//MICU"),
                (datetime(2023, 2, 1), "TRANSFER_TO//ED//Emergency Department"),
                (datetime(2023, 3, 1), "HOSPITAL_ADMISSION//URGENT//EMERGENCY ROOM"),
                # Event after prediction_time — must NOT be counted.
                (datetime(2025, 1, 1), "ICU_ADMISSION//SICU"),
            ]
        },
    )
    summary = compute_patient_clinical_summary(42, datetime(2024, 1, 1), cohort)
    assert summary["n_hospital_admissions"] == 2
    assert summary["n_icu_admissions"] == 1
    assert summary["n_ed_visits"] == 1


def test_compute_patient_clinical_summary_flags_chronic_conditions_from_icd10(
    tmp_path: Path,
) -> None:
    cohort = _make_meds_cohort(
        tmp_path,
        {
            7: [
                (datetime(2023, 1, 1), "DIAGNOSIS//ICD//10//E1165"),  # diabetes
                (datetime(2023, 1, 2), "DIAGNOSIS//ICD//10//N183"),   # CKD
                (datetime(2023, 1, 3), "DIAGNOSIS//ICD//10//I509"),   # CHF
                (datetime(2023, 1, 4), "DIAGNOSIS//ICD//10//J449"),   # COPD
                (datetime(2023, 1, 5), "DIAGNOSIS//ICD//10//Z5189"),  # non-Charlson
            ]
        },
    )
    summary = compute_patient_clinical_summary(7, datetime(2024, 1, 1), cohort)
    conds = set(summary["chronic_conditions"])
    assert "Diabetes mellitus" in conds
    assert "Chronic kidney disease" in conds
    assert "Congestive heart failure" in conds
    assert "Chronic pulmonary disease / COPD" in conds
    assert summary["chronic_condition_count"] == 4
    # Non-matching Z-code must not create a flag.
    assert not any("Z" in c for c in conds)


def test_timeline_renderer_includes_clinical_summary_when_provided(tmp_path: Path) -> None:
    codes_fp = _save_codes_parquet(tmp_path, ["LAB//E1"], ["Hemoglobin"])
    cohort = _make_meds_cohort(
        tmp_path,
        {9: [(datetime(2024, 1, 1), "LAB//E1")]},
    )
    renderer = PatientTimelineRenderer(codes_parquet=codes_fp, max_events=5)
    summary = {
        "n_hospital_admissions": 3,
        "n_icu_admissions": 1,
        "n_ed_visits": 2,
        "chronic_conditions": ["Diabetes mellitus", "Chronic kidney disease"],
        "chronic_condition_count": 2,
    }
    text = renderer.render(
        9, datetime(2024, 1, 2), cohort, clinical_summary=summary,
    )
    assert "CLINICAL SUMMARY:" in text
    assert "3 hospital admission(s)" in text
    assert "1 ICU stay(s)" in text
    assert "2 ED visit(s)" in text
    assert "Diabetes mellitus" in text
    assert "Chronic kidney disease" in text
    # Without a summary, the marker must be absent.
    text_plain = renderer.render(9, datetime(2024, 1, 2), cohort)
    assert "CLINICAL SUMMARY:" not in text_plain


def test_clean_lab_description_strips_loinc_cruft() -> None:
    cases = {
        "Carbon dioxide, total [Moles/volume] in Blood by calculation": "Carbon dioxide, total",
        "Lactate [Moles/volume] in Blood": "Lactate",
        "pH of Blood": "pH",
        "Oxygen [Partial pressure] in Blood": "Oxygen",
        "Urea nitrogen [Mass/volume] in Serum or Plasma": "Urea nitrogen",
        "Leukocytes [#/volume] in Blood by Automated count": "Leukocytes",
        "Sodium [Moles/volume] in Serum or Plasma": "Sodium",
        # Already-clean strings must be untouched.
        "Hemoglobin": "Hemoglobin",
        "Anion gap 4 in Serum or Plasma": "Anion gap 4",
    }
    for raw, expected in cases.items():
        assert _clean_lab_description(raw) == expected, f"{raw!r} -> {_clean_lab_description(raw)!r}"


def test_patient_narrative_combines_demographics_and_clinical_summary() -> None:
    demographics = {"gender": "F", "race": "WHITE", "birth_time": datetime(1958, 1, 1)}
    summary = {
        "n_hospital_admissions": 1,
        "n_icu_admissions": 1,
        "n_ed_visits": 1,
        "chronic_conditions": [],
        "chronic_condition_count": 0,
    }
    text = _render_patient_narrative(demographics, datetime(2026, 6, 1), summary)
    assert text is not None
    assert "68-year-old" in text
    assert "White" in text
    assert "woman" in text
    assert "1 hospital admission" in text
    assert "1 ICU stay" in text
    assert "1 ED visit" in text
    assert "No chronic conditions detected" in text
    # Explicitly prose — no all-caps section headers.
    assert "PATIENT:" not in text
    assert "CLINICAL SUMMARY:" not in text


def test_patient_narrative_inlines_chronic_conditions_when_present() -> None:
    demographics = {"gender": "M", "birth_time": datetime(1960, 1, 1)}
    summary = {
        "n_hospital_admissions": 0,
        "n_icu_admissions": 0,
        "n_ed_visits": 0,
        "chronic_conditions": ["Diabetes mellitus", "Chronic kidney disease"],
        "chronic_condition_count": 2,
    }
    text = _render_patient_narrative(demographics, datetime(2024, 1, 1), summary)
    assert text is not None
    assert "History notable for Diabetes mellitus, Chronic kidney disease" in text


def test_categorical_renderer_partitions_events_by_category(tmp_path: Path) -> None:
    codes_fp = _save_codes_parquet(
        tmp_path,
        [
            "DIAGNOSIS//ICD//10//E11",
            "MEDICATION//METFORMIN",
            "PROCEDURE//ICD//10//0T1807C",
            "LAB//50811//g/dL",
            "HOSPITAL_ADMISSION//EW EMER.",
            "ICU_ADMISSION//MICU",
        ],
        [
            "Type 2 diabetes mellitus",
            "METFORMIN",
            "Bypass ureters",
            "Hemoglobin",
            "EW Emergency Admission",
            "Medical ICU",
        ],
    )
    cohort = _make_meds_cohort(
        tmp_path,
        {
            1: [
                (datetime(2024, 1, 1), "HOSPITAL_ADMISSION//EW EMER."),
                (datetime(2024, 1, 1), "DIAGNOSIS//ICD//10//E11"),
                (datetime(2024, 1, 1), "MEDICATION//METFORMIN"),
                (datetime(2024, 1, 2), "PROCEDURE//ICD//10//0T1807C"),
                (datetime(2024, 1, 2), "LAB//50811//g/dL", 9.5),
                (datetime(2024, 1, 2), "ICU_ADMISSION//MICU"),
            ],
        },
    )
    renderer = PatientTimelineRenderer(codes_parquet=codes_fp, max_events=100)
    text = renderer.render_categorical(1, datetime(2024, 1, 3), cohort)
    assert "Diagnoses (from history):" in text
    assert "Active medications (recent):" in text
    assert "Recent procedures:" in text
    assert "Recent labs (most recent value):" in text
    assert "Type 2 diabetes mellitus" in text
    assert "METFORMIN" in text
    assert "Bypass ureters" in text
    assert "Hemoglobin: 9.50 g/dL" in text
    # Admission/transfer codes are suppressed — they live in CLINICAL SUMMARY.
    assert "EW Emergency Admission" not in text
    assert "Medical ICU" not in text


def test_categorical_renderer_dedupes_within_category_and_keeps_latest_lab(
    tmp_path: Path,
) -> None:
    codes_fp = _save_codes_parquet(
        tmp_path,
        ["DIAGNOSIS//ICD//10//E11", "LAB//50811//g/dL"],
        ["Type 2 diabetes mellitus", "Hemoglobin"],
    )
    cohort = _make_meds_cohort(
        tmp_path,
        {
            2: [
                (datetime(2024, 1, 1), "DIAGNOSIS//ICD//10//E11"),
                (datetime(2024, 1, 2), "DIAGNOSIS//ICD//10//E11"),
                (datetime(2024, 1, 1), "LAB//50811//g/dL", 7.0),
                (datetime(2024, 1, 2), "LAB//50811//g/dL", 12.0),
            ],
        },
    )
    renderer = PatientTimelineRenderer(codes_parquet=codes_fp, max_events=100)
    text = renderer.render_categorical(2, datetime(2024, 1, 3), cohort)
    # Diagnosis listed once despite two occurrences.
    assert text.count("- Type 2 diabetes mellitus") == 1
    # Lab listed once, showing the most recent value (12, not 7).
    assert "Hemoglobin: 12 g/dL" in text
    assert "Hemoglobin: 7" not in text


def test_categorical_renderer_caps_per_category(tmp_path: Path) -> None:
    codes_fp = _save_codes_parquet(
        tmp_path,
        [f"DIAGNOSIS//ICD//10//D{i}" for i in range(10)],
        [f"Diagnosis {i}" for i in range(10)],
    )
    cohort = _make_meds_cohort(
        tmp_path,
        {
            3: [
                (datetime(2024, 1, 1, 0, i), f"DIAGNOSIS//ICD//10//D{i}") for i in range(10)
            ],
        },
    )
    renderer = PatientTimelineRenderer(codes_parquet=codes_fp, max_events=100)
    text = renderer.render_categorical(
        3, datetime(2024, 1, 2), cohort, max_diagnoses=3
    )
    # Cap retains the last 3 (most recent) only.
    assert "Diagnosis 9" in text
    assert "Diagnosis 7" in text
    assert "Diagnosis 6" not in text
    assert text.count("- Diagnosis") == 3


# ---------------------------------------------------------------------------
# T16–T19, T24: aggregation / bootstrap
# ---------------------------------------------------------------------------


def _verdicts_df(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal verdicts df accepted by summarize_winrates."""
    defaults = {
        "winner_position": "A",
        "confidence": 1.0,
        "rationale": "",
        "raw_response": "{}",
        "model": "fake",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "target_position": "A",
        "target_doc_id": 0,
        "other_doc_id": 0,
        "other_rank": None,
        "other_source_subject_id": None,
        "anchor_row_idx": 0,
    }
    merged = [{**defaults, **r} for r in rows]
    return pl.DataFrame(merged)


def test_within_patient_averaging_before_aggregation() -> None:
    # Patient 1 contributes (win, loss) = 0.5; patient 2 contributes (win) = 1.0.
    df = _verdicts_df(
        [
            {"pair_id": "p1", "family": "F1", "anchor_subject_id": 1, "anchor_label": 1, "target_won": True},
            {"pair_id": "p2", "family": "F1", "anchor_subject_id": 1, "anchor_label": 1, "target_won": False},
            {"pair_id": "p3", "family": "F1", "anchor_subject_id": 2, "anchor_label": 1, "target_won": True},
        ]
    )
    summary = summarize_winrates(df, n_bootstrap=50, seed=0)
    f1 = summary.filter(pl.col("family") == "F1").row(0, named=True)
    # Naive pair-mean would be 2/3 ≈ 0.667; correct mean-of-means = (0.5 + 1.0) / 2 = 0.75.
    assert abs(f1["target_preferred_rate"] - 0.75) < 1e-9


def test_summarize_winrates_point_estimate_matches_hand_computed() -> None:
    # 4 patients, one pair each. target_won: T, T, F, T → 3/4 = 0.75.
    df = _verdicts_df(
        [
            {"pair_id": f"p{i}", "family": "F1", "anchor_subject_id": i, "anchor_label": 0, "target_won": w}
            for i, w in enumerate([True, True, False, True], start=1)
        ]
    )
    summary = summarize_winrates(df, n_bootstrap=50, seed=0)
    f1 = summary.filter(pl.col("family") == "F1").row(0, named=True)
    assert abs(f1["target_preferred_rate"] - 0.75) < 1e-9
    assert f1["n_patients"] == 4


def test_summarize_winrates_bootstrap_ci_covers_point_estimate() -> None:
    df = _verdicts_df(
        [
            {"pair_id": f"p{i}", "family": "F1", "anchor_subject_id": i, "anchor_label": 0, "target_won": bool(i % 2)}
            for i in range(1, 21)
        ]
    )
    summary = summarize_winrates(df, n_bootstrap=500, seed=0)
    f1 = summary.filter(pl.col("family") == "F1").row(0, named=True)
    assert f1["ci_low"] <= f1["target_preferred_rate"] <= f1["ci_high"]


def test_summarize_winrates_half_credit_ties_matches_hand_computed() -> None:
    """Each tie counts as 0.5 of a win, ie rate = (wins + 0.5*ties) / n_pairs.

    With 3 wins, 6 losses, 91 ties across distinct anchors (one pair per
    patient, so the per-patient mean equals the per-pair value), the
    headline rate must equal (3 + 0.5 * 91) / 100 = 0.485.
    """
    rows: list[dict] = []
    anchor = 1
    for _ in range(3):  # wins
        rows.append({
            "pair_id": f"p{anchor}", "family": "F1", "anchor_subject_id": anchor,
            "anchor_label": 0, "target_won": True, "winner_position": "A",
        })
        anchor += 1
    for _ in range(6):  # losses
        rows.append({
            "pair_id": f"p{anchor}", "family": "F1", "anchor_subject_id": anchor,
            "anchor_label": 0, "target_won": False, "winner_position": "B",
        })
        anchor += 1
    for _ in range(91):  # ties
        rows.append({
            "pair_id": f"p{anchor}", "family": "F1", "anchor_subject_id": anchor,
            "anchor_label": 0, "target_won": None, "winner_position": "tie",
        })
        anchor += 1
    df = _verdicts_df(rows)

    half = summarize_winrates(df, n_bootstrap=200, seed=0, invalid_policy="half_credit_ties")
    half_row = half.filter(pl.col("family") == "F1").row(0, named=True)
    assert abs(half_row["target_preferred_rate"] - 0.485) < 1e-9
    # SE/CI are well-defined under half-credit; bootstrap shouldn't collapse to NaN.
    assert half_row["n_patients"] == 100
    assert not math.isnan(half_row["standard_error"])
    assert not math.isnan(half_row["ci_low"])
    assert not math.isnan(half_row["ci_high"])
    assert half_row["ci_low"] <= half_row["target_preferred_rate"] <= half_row["ci_high"]

    # Cross-check against the other policies on the same data.
    drop = summarize_winrates(df, n_bootstrap=0, seed=0, invalid_policy="drop")
    drop_row = drop.filter(pl.col("family") == "F1").row(0, named=True)
    assert abs(drop_row["target_preferred_rate"] - (3.0 / 9.0)) < 1e-9
    assert drop_row["n_patients"] == 9

    counted = summarize_winrates(df, n_bootstrap=0, seed=0, invalid_policy="count_as_loss")
    counted_row = counted.filter(pl.col("family") == "F1").row(0, named=True)
    assert abs(counted_row["target_preferred_rate"] - 0.03) < 1e-9
    assert counted_row["n_patients"] == 100


def test_summarize_winrates_drop_vs_count_as_loss_policy() -> None:
    df = _verdicts_df(
        [
            {"pair_id": "p1", "family": "F1", "anchor_subject_id": 1, "anchor_label": 0, "target_won": True,
             "winner_position": "A"},
            {"pair_id": "p2", "family": "F1", "anchor_subject_id": 2, "anchor_label": 0, "target_won": None,
             "winner_position": "tie"},
        ]
    )
    drop = summarize_winrates(df, n_bootstrap=50, seed=0, invalid_policy="drop")
    counted = summarize_winrates(df, n_bootstrap=50, seed=0, invalid_policy="count_as_loss")
    drop_row = drop.filter(pl.col("family") == "F1").row(0, named=True)
    counted_row = counted.filter(pl.col("family") == "F1").row(0, named=True)
    # drop: 1 patient → 1.0. count_as_loss: 2 patients → 0.5.
    assert abs(drop_row["target_preferred_rate"] - 1.0) < 1e-9
    assert abs(counted_row["target_preferred_rate"] - 0.5) < 1e-9
    assert drop_row["n_invalid"] == 1
    assert counted_row["n_invalid"] == 1
    # The tie contributes to n_ties but not to the error categories.
    assert drop_row["n_ties"] == 1
    assert drop_row["n_api_errors"] == 0
    assert drop_row["n_parse_errors"] == 0
    assert drop_row["n_client_init_errors"] == 0
    assert drop_row["n_other_invalid"] == 0


def test_summarize_winrates_splits_invalid_into_labeled_error_columns() -> None:
    # One of each failure mode, plus one valid row. target_won=None for invalids.
    df = _verdicts_df(
        [
            {"pair_id": "valid", "family": "F1", "anchor_subject_id": 1, "anchor_label": 0,
             "target_won": True, "winner_position": "A", "rationale": "ok"},
            {"pair_id": "tie", "family": "F1", "anchor_subject_id": 2, "anchor_label": 0,
             "target_won": None, "winner_position": "tie", "rationale": "equally relevant"},
            {"pair_id": "api", "family": "F1", "anchor_subject_id": 3, "anchor_label": 0,
             "target_won": None, "winner_position": "invalid", "rationale": "api error: 500 server error"},
            {"pair_id": "parse", "family": "F1", "anchor_subject_id": 4, "anchor_label": 0,
             "target_won": None, "winner_position": "invalid", "rationale": "parse error: bad JSON"},
            {"pair_id": "init", "family": "F1", "anchor_subject_id": 5, "anchor_label": 0,
             "target_won": None, "winner_position": "invalid",
             "rationale": "openai client init failed: missing OPENAI_API_KEY"},
            {"pair_id": "other", "family": "F1", "anchor_subject_id": 6, "anchor_label": 0,
             "target_won": None, "winner_position": "invalid", "rationale": "something else"},
        ]
    )
    summary = summarize_winrates(df, n_bootstrap=50, seed=0)
    row = summary.filter(pl.col("family") == "F1").row(0, named=True)
    assert row["n_invalid"] == 5
    assert row["n_ties"] == 1
    assert row["n_api_errors"] == 1
    assert row["n_parse_errors"] == 1
    assert row["n_client_init_errors"] == 1
    assert row["n_other_invalid"] == 1
    # Error-category counts must sum to n_invalid.
    assert (
        row["n_ties"]
        + row["n_api_errors"]
        + row["n_parse_errors"]
        + row["n_client_init_errors"]
        + row["n_other_invalid"]
        == row["n_invalid"]
    )


def test_summarize_winrates_standard_error_matches_bootstrap_std() -> None:
    # With identical per-patient values, bootstrap variance must be 0.
    df = _verdicts_df(
        [
            {"pair_id": f"p{i}", "family": "F1", "anchor_subject_id": i, "anchor_label": 0, "target_won": True}
            for i in range(1, 11)
        ]
    )
    summary = summarize_winrates(df, n_bootstrap=500, seed=0)
    f1 = summary.filter(pl.col("family") == "F1").row(0, named=True)
    assert f1["standard_error"] == pytest.approx(0.0, abs=1e-9)
    assert f1["target_preferred_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# T20–T21, T31: human validation subset
# ---------------------------------------------------------------------------


def test_human_validation_subset_strips_target_and_position_columns(tmp_path: Path) -> None:
    ds_path = _save_retrieval_ds(tmp_path, titles=[f"t{i}" for i in range(5)])
    from datasets import load_from_disk

    ds = load_from_disk(str(ds_path))
    df = _verdicts_df(
        [
            {"pair_id": f"p{i}", "family": "F1", "anchor_subject_id": i, "anchor_label": 0,
             "target_won": True, "target_doc_id": 100, "other_doc_id": 101, "target_position": "A"}
            for i in range(1, 11)
        ]
    )
    subset = build_human_validation_subset(
        df, n=5, seed=0, retrieval_ds=ds, doc_id_to_row={100 + i: i for i in range(5)}
    )
    banned = {"target_doc_id", "target_position", "target_won", "winner_position", "model", "raw_response"}
    assert banned.isdisjoint(subset.columns)


def test_human_validation_subset_preserves_pair_id_for_rejoining(tmp_path: Path) -> None:
    ds_path = _save_retrieval_ds(tmp_path, titles=[f"t{i}" for i in range(5)])
    from datasets import load_from_disk

    ds = load_from_disk(str(ds_path))
    df = _verdicts_df(
        [
            {"pair_id": f"p{i}", "family": "F1", "anchor_subject_id": i, "anchor_label": 0,
             "target_won": True, "target_doc_id": 100 + (i % 5), "other_doc_id": 100 + ((i + 1) % 5),
             "target_position": "A"}
            for i in range(1, 11)
        ]
    )
    subset = build_human_validation_subset(
        df, n=5, seed=0, retrieval_ds=ds, doc_id_to_row={100 + i: i for i in range(5)}
    )
    assert "pair_id" in subset.columns
    assert subset["pair_id"].n_unique() == subset.height


def test_human_validation_subset_keeps_doc_titles_and_timeline_but_not_target_position(
    tmp_path: Path,
) -> None:
    ds_path = _save_retrieval_ds(tmp_path, titles=[f"t{i}" for i in range(5)])
    from datasets import load_from_disk

    ds = load_from_disk(str(ds_path))
    df = _verdicts_df(
        [
            {"pair_id": f"p{i}", "family": "F1", "anchor_subject_id": i, "anchor_label": 0,
             "target_won": True, "target_doc_id": 100 + (i % 5), "other_doc_id": 100 + ((i + 1) % 5),
             "target_position": "A", "patient_timeline": f"timeline-{i}"}
            for i in range(1, 11)
        ]
    )
    subset = build_human_validation_subset(
        df, n=5, seed=0, retrieval_ds=ds, doc_id_to_row={100 + i: i for i in range(5)}
    )
    # Titles must appear (rater can read anyway), but target_position must not.
    assert "doc_a_title" in subset.columns or "doc_a_text" in subset.columns
    assert "target_position" not in subset.columns


# ---------------------------------------------------------------------------
# T22: OpenAIJudge error handling
# ---------------------------------------------------------------------------


def test_openai_judge_never_raises_on_parser_failure() -> None:
    class _MockClient:
        class chat:  # noqa: N801 - matching openai API surface
            class completions:  # noqa: N801
                @staticmethod
                def create(**_kw):
                    raise RuntimeError("boom from API")

    judge = OpenAIJudge(model="gpt-4o-mini", client=_MockClient())
    v = judge.judge("sys", "user", seed=0)
    assert v.winner_position == "invalid"
    assert v.target_won is None


# ---------------------------------------------------------------------------
# T23: FakeJudge protocol conformance
# ---------------------------------------------------------------------------


def test_fake_judge_implements_judge_protocol() -> None:
    judge = FakeJudge.always_A()
    assert isinstance(judge, Judge)


# ---------------------------------------------------------------------------
# T25–T30: per-patient rollup
# ---------------------------------------------------------------------------


def _rollup_inputs(tmp_path: Path, with_title: bool = True, with_race: bool = True) -> dict:
    artifacts = _make_artifacts(n_patients=4, k=3, labels=[0, 1, 0, 1], top1_doc_ids=[100, 101, 102, 103])
    schema = _make_val_schema([10, 20, 30, 40])
    pairs = [
        JudgePair(pair_id=f"p{fam}-{i}", family=fam, anchor_row_idx=i,
                  anchor_subject_id=int(schema["subject_id"][i]), anchor_label=int(artifacts["targets"][i]),
                  target_doc_id=int(artifacts["doc_ids"][i, 0, 0]),
                  other_doc_id=int(artifacts["doc_ids"][i, 0, 0]) + 1,
                  target_position="A", other_rank=(2 if fam == "F2" else None))
        for i in range(4)
        for fam in ("F1", "F2", "F3", "F4")
    ]
    verdicts = _verdicts_df(
        [
            {"pair_id": p.pair_id, "family": p.family, "anchor_subject_id": p.anchor_subject_id,
             "anchor_label": p.anchor_label, "target_won": True,
             "target_doc_id": p.target_doc_id, "other_doc_id": p.other_doc_id,
             "target_position": "A", "other_rank": p.other_rank, "anchor_row_idx": p.anchor_row_idx,
             "confidence": 0.9, "rationale": "because", "winner_position": "A"}
            for p in pairs
        ]
    )

    titles = [f"book-{i}" for i in range(200)]
    extra = {}
    if with_title:
        ds_cols = {"title": titles}
    else:
        ds_cols = {"other_col": titles}
    ds_path = tmp_path / "retrieval_db"
    Dataset.from_dict(
        {**ds_cols, "doc_text": [f"body-{i}" for i in range(200)], "doc_ids": list(range(100, 300))}
    ).save_to_disk(str(ds_path))
    from datasets import load_from_disk

    ds = load_from_disk(str(ds_path))
    doc_id_to_row = {100 + i: i for i in range(200)}
    codes_fp = _save_codes_parquet(tmp_path, ["E1"], ["desc"])

    dem_cols = {"subject_id": [10, 20, 30, 40], "gender": ["M", "F", "M", "F"],
                "birth_time": [datetime(1950, 1, 1)] * 4}
    if with_race:
        dem_cols["race"] = ["WHITE", "BLACK/AFRICAN AMERICAN", "ASIAN", "HISPANIC/LATINO"]
    demographics = pl.DataFrame(dem_cols)

    return dict(
        pairs=pairs,
        verdicts=verdicts,
        logits=artifacts["logits"],
        targets=artifacts["targets"],
        artifacts=artifacts,
        timeline_renderer=PatientTimelineRenderer(codes_parquet=codes_fp),
        val_schema=schema,
        demographics=demographics,
        retrieval_ds=ds,
        doc_id_to_row=doc_id_to_row,
    )


def test_build_per_patient_rollup_one_row_per_patient_with_all_families(tmp_path: Path) -> None:
    inputs = _rollup_inputs(tmp_path)
    df = build_per_patient_rollup(**inputs)
    assert df.height == 4  # one row per patient
    for f in ("F1", "F2", "F3", "F4"):
        assert f"{f}_target_won" in df.columns


def test_build_per_patient_rollup_merges_multiple_pairs_into_mean_target_won(tmp_path: Path) -> None:
    inputs = _rollup_inputs(tmp_path)
    # Duplicate pairs for F1 with mixed outcomes to test within-patient mean.
    extra = inputs["verdicts"].filter(pl.col("family") == "F1").with_columns(
        pl.lit(False).alias("target_won"),
        pl.col("pair_id") + "-dup",
    )
    inputs["verdicts"] = pl.concat([inputs["verdicts"], extra])
    # Extend pairs list similarly.
    inputs["pairs"] = list(inputs["pairs"]) + [
        JudgePair(pair_id=p.pair_id + "-dup", family=p.family, anchor_row_idx=p.anchor_row_idx,
                  anchor_subject_id=p.anchor_subject_id, anchor_label=p.anchor_label,
                  target_doc_id=p.target_doc_id, other_doc_id=p.other_doc_id,
                  target_position=p.target_position, other_rank=p.other_rank)
        for p in inputs["pairs"] if p.family == "F1"
    ]
    df = build_per_patient_rollup(**inputs)
    # Each patient had (T, F) for F1 → mean should be 0.5.
    assert all(abs(v - 0.5) < 1e-9 for v in df["F1_target_won"].to_list())


def test_per_patient_rollup_joins_doc_metadata_columns(tmp_path: Path) -> None:
    inputs = _rollup_inputs(tmp_path, with_title=True)
    df = build_per_patient_rollup(**inputs)
    assert "target_doc_title" in df.columns
    # Titles aligned to correct doc rows (row 0 → book-0 via doc_id 100).
    row = df.filter(pl.col("anchor_subject_id") == 10).row(0, named=True)
    assert row["target_doc_title"] == "book-0"


def test_per_patient_rollup_joins_demographics(tmp_path: Path) -> None:
    inputs = _rollup_inputs(tmp_path)
    df = build_per_patient_rollup(**inputs)
    by_sid = {r["anchor_subject_id"]: r for r in df.iter_rows(named=True)}
    assert by_sid[10]["gender"] == "M"
    assert by_sid[20]["gender"] == "F"
    assert by_sid[20]["race"] == "BLACK/AFRICAN AMERICAN"


def test_per_patient_rollup_skips_missing_doc_metadata_columns(tmp_path: Path) -> None:
    inputs = _rollup_inputs(tmp_path, with_title=False)
    df = build_per_patient_rollup(**inputs, doc_metadata_columns=("title",))
    # 'title' is not present on the ds; column must simply be absent (no crash).
    assert "target_doc_title" not in df.columns


def test_per_patient_rollup_includes_clinical_summary_columns(tmp_path: Path) -> None:
    inputs = _rollup_inputs(tmp_path)
    summaries = {
        10: {"n_hospital_admissions": 2, "n_icu_admissions": 1, "n_ed_visits": 0,
             "chronic_conditions": ["Diabetes mellitus"], "chronic_condition_count": 1},
        20: {"n_hospital_admissions": 5, "n_icu_admissions": 2, "n_ed_visits": 3,
             "chronic_conditions": [], "chronic_condition_count": 0},
    }
    df = build_per_patient_rollup(**inputs, clinical_summaries_by_subject_id=summaries)
    by_sid = {r["anchor_subject_id"]: r for r in df.iter_rows(named=True)}
    assert by_sid[10]["n_hospital_admissions"] == 2
    assert by_sid[10]["n_icu_admissions"] == 1
    assert by_sid[10]["chronic_conditions"] == "Diabetes mellitus"
    assert by_sid[10]["chronic_condition_count"] == 1
    assert by_sid[20]["n_ed_visits"] == 3
    assert by_sid[20]["chronic_conditions"] == ""
    # Patients without a summary (30, 40) must still appear, with None columns.
    assert by_sid[30]["n_hospital_admissions"] is None
    assert by_sid[30]["chronic_conditions"] is None


# ---------------------------------------------------------------------------
# T27: workbook writer
# ---------------------------------------------------------------------------


def test_write_results_workbook_produces_all_four_sheets(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "results.xlsx"
    empty = pl.DataFrame({"col": [1]})
    write_results_workbook(
        path,
        family_winrates=empty,
        per_patient=empty,
        pairs_verdicts=empty,
        human_validation=empty,
    )
    assert path.is_file()
    wb = openpyxl.load_workbook(path)
    assert set(wb.sheetnames) == {"family_winrates", "per_patient_results", "pairs_verdicts", "human_validation"}
