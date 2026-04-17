"""Tests for the LLM-as-a-judge retrieval-relevance evaluation module.

All tests use the in-memory :class:`FakeJudge` + tiny polars / HF Dataset
fixtures. No live OpenAI call is made. Structure mirrors the 31-item
spec in ``D3_plan.md`` § "Test layout".
"""

from __future__ import annotations

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
    build_human_validation_subset,
    build_pairs,
    build_per_patient_rollup,
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


def _make_meds_cohort(tmp_path: Path, subject_events: dict[int, list[tuple[datetime, str]]]) -> Path:
    """Write a one-shard cohort with the given (time, code) per subject."""
    cohort = tmp_path / "meds_cohort"
    data_dir = cohort / "data" / "train"
    data_dir.mkdir(parents=True)
    rows = []
    for sid, events in subject_events.items():
        for t, code in events:
            rows.append({"subject_id": sid, "time": t, "code": code})
    pl.DataFrame(rows).write_parquet(data_dir / "shard_0.parquet")
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
    codes_fp = _save_codes_parquet(tmp_path, ["E1", "E2", "E3", "E4", "E5"], [None] * 5)
    cohort = _make_meds_cohort(
        tmp_path,
        {
            42: [
                (datetime(2024, 1, 1), "E1"),
                (datetime(2024, 1, 2), "E2"),
                (datetime(2024, 1, 3), "E3"),
                (datetime(2024, 1, 4), "E4"),
                (datetime(2024, 1, 5), "E5"),
            ]
        },
    )
    renderer = PatientTimelineRenderer(codes_parquet=codes_fp, max_events=3)
    text = renderer.render(42, datetime(2024, 1, 4), cohort)
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines == ["E2", "E3", "E4"]


def test_timeline_renderer_annotates_codes_with_descriptions_from_parquet(tmp_path: Path) -> None:
    codes_fp = _save_codes_parquet(
        tmp_path,
        ["DIAG//I509", "LAB//50811", "MEDS_BIRTH"],
        ["Heart failure, unspecified", "Hemoglobin", None],
    )
    cohort = _make_meds_cohort(
        tmp_path,
        {
            7: [
                (datetime(2023, 1, 1), "MEDS_BIRTH"),
                (datetime(2024, 1, 1), "DIAG//I509"),
                (datetime(2024, 1, 2), "LAB//50811"),
            ]
        },
    )
    renderer = PatientTimelineRenderer(codes_parquet=codes_fp, max_events=50)
    text = renderer.render(7, datetime(2025, 1, 1), cohort)
    assert "DIAG//I509 — Heart failure, unspecified" in text
    assert "LAB//50811 — Hemoglobin" in text
    # Null description → bare code, no em-dash.
    assert "MEDS_BIRTH\n" in text or text.endswith("MEDS_BIRTH")
    assert "MEDS_BIRTH —" not in text


def test_codes_parquet_must_be_one_to_one(tmp_path: Path) -> None:
    codes_fp = _save_codes_parquet(tmp_path, ["DUP_CODE", "DUP_CODE", "OTHER"], ["d1", "d2", "d3"])
    with pytest.raises(ValueError, match="DUP_CODE"):
        PatientTimelineRenderer(codes_parquet=codes_fp)


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
