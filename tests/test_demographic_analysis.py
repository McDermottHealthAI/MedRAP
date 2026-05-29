from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from datasets import Dataset

from medrap.demographic_analysis import (
    AGE_BIN_ORDER,
    RACE_BIN_ORDER,
    UNKNOWN_BIN,
    LDATopicProvider,
    StaticMappingProvider,
    TitleKeywordProvider,
    _build_doc_id_to_row_map,
    _resolve_doc_row_index,
    _softmax,
    aggregate_race,
    bin_age,
    build_comorbidity_keyword_table,
    build_keyword_demographic_table,
    build_patient_demographic_frame,
    build_pearson_residual_table,
    extract_val_schema,
    load_subject_demographics,
    render_demographic_heatmaps,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_title_keyword_provider_resolves_doc_ids_column(tmp_path: Path) -> None:
    dataset_path = tmp_path / "retrieval_db"
    Dataset.from_dict(
        {
            "title": ["Cardiology", "Obstetrics", "Neurology"],
            "doc_ids": [101, 42, 7],
        }
    ).save_to_disk(str(dataset_path))

    provider = TitleKeywordProvider(dataset_path)

    assert provider.keywords_for(42) == [("Obstetrics", 1.0)]
    assert provider.keywords_for(101) == [("Cardiology", 1.0)]
    assert provider.keywords_for(7) == [("Neurology", 1.0)]


def test_title_keyword_provider_falls_back_to_row_indices_when_doc_ids_absent(tmp_path: Path) -> None:
    dataset_path = tmp_path / "retrieval_db"
    Dataset.from_dict(
        {
            "title": ["A", "B"],
        }
    ).save_to_disk(str(dataset_path))

    provider = TitleKeywordProvider(dataset_path)

    assert provider.keywords_for(0) == [("A", 1.0)]
    assert provider.keywords_for(1) == [("B", 1.0)]


def test_title_keyword_provider_rejects_duplicate_doc_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "retrieval_db"
    Dataset.from_dict(
        {
            "title": ["A", "B"],
            "doc_ids": [5, 5],
        }
    ).save_to_disk(str(dataset_path))

    with pytest.raises(ValueError, match="duplicate doc_ids"):
        TitleKeywordProvider(dataset_path)


def test_title_keyword_provider_allows_string_doc_ids_and_row_index_fallback(tmp_path: Path) -> None:
    dataset_path = tmp_path / "retrieval_db"
    Dataset.from_dict(
        {
            "title": ["A", "B", "C"],
            "doc_ids": ["doc_a", "doc_b", "doc_c"],
        }
    ).save_to_disk(str(dataset_path))

    provider = TitleKeywordProvider(dataset_path)

    # Direct id lookup path.
    assert provider.keywords_for("doc_b") == [("B", 1.0)]
    # Backward-compatible fallback for artifacts storing row indices.
    assert provider.keywords_for(2) == [("C", 1.0)]


def test_title_keyword_provider_rejects_dataset_without_title(tmp_path: Path) -> None:
    dataset_path = tmp_path / "retrieval_db"
    Dataset.from_dict({"content": ["hello"]}).save_to_disk(str(dataset_path))

    with pytest.raises(ValueError, match="no 'title' column"):
        TitleKeywordProvider(dataset_path)


def test_title_keyword_provider_exposes_sorted_unique_vocab(tmp_path: Path) -> None:
    dataset_path = tmp_path / "retrieval_db"
    Dataset.from_dict({"title": ["Neurology", "Cardiology", "Cardiology"]}).save_to_disk(str(dataset_path))

    provider = TitleKeywordProvider(dataset_path)
    assert provider.vocab == ["Cardiology", "Neurology"]


def test_aggregate_race_maps_known_values_to_buckets() -> None:
    assert aggregate_race("WHITE") == "White"
    assert aggregate_race("BLACK/AFRICAN AMERICAN") == "Black"
    assert aggregate_race("ASIAN - CHINESE") == "Asian"
    assert aggregate_race("HISPANIC/LATINO") == "Hispanic/Latino"
    assert aggregate_race("AMERICAN INDIAN/ALASKA NATIVE") == "American Indian/Alaska Native"


def test_aggregate_race_handles_none_and_unknown() -> None:
    assert aggregate_race(None) == "Other/Unknown"
    assert aggregate_race("NOT A REAL VALUE") == "Other/Unknown"


def test_aggregate_race_strips_surrounding_whitespace() -> None:
    assert aggregate_race("  WHITE  ") == "White"


def test_bin_age_covers_all_age_bins() -> None:
    assert bin_age(5.0) == "0-18"
    assert bin_age(18.0) == "18-30"
    assert bin_age(29.9) == "18-30"
    assert bin_age(30.0) == "30-45"
    assert bin_age(50.0) == "45-60"
    assert bin_age(70.0) == "60-75"
    assert bin_age(100.0) == "75+"


def test_bin_age_returns_unknown_for_none_nan_and_inf() -> None:
    assert bin_age(None) == UNKNOWN_BIN
    assert bin_age(float("nan")) == UNKNOWN_BIN
    assert bin_age(float("inf")) == UNKNOWN_BIN


def test_bin_age_returns_unknown_for_negative_age() -> None:
    assert bin_age(-1.0) == UNKNOWN_BIN


def test_age_bin_order_ends_with_unknown_bucket() -> None:
    assert AGE_BIN_ORDER[-1] == UNKNOWN_BIN
    assert AGE_BIN_ORDER[0] == "0-18"


def test_race_bin_order_contains_expected_buckets() -> None:
    assert "White" in RACE_BIN_ORDER
    assert "Other/Unknown" in RACE_BIN_ORDER
    assert len(RACE_BIN_ORDER) == len(set(RACE_BIN_ORDER))


def test_static_mapping_provider_returns_keywords_and_sorted_vocab() -> None:
    provider = StaticMappingProvider(
        [
            [("cardio", 1.0)],
            [("neuro", 0.7), ("onco", 0.3)],
            [("cardio", 0.5), ("neuro", 0.5)],
        ]
    )

    assert provider.keywords_for(0) == [("cardio", 1.0)]
    assert provider.keywords_for(1) == [("neuro", 0.7), ("onco", 0.3)]
    assert provider.vocab == ["cardio", "neuro", "onco"]


def test_static_mapping_provider_returns_defensive_copy_of_entries() -> None:
    provider = StaticMappingProvider([[("a", 1.0)]])
    result = provider.keywords_for(0)
    result.append(("extra", 0.0))

    assert provider.keywords_for(0) == [("a", 1.0)]


def test_softmax_sums_to_one_along_last_axis() -> None:
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    out = _softmax(x, axis=-1)

    assert out.shape == x.shape
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(2))


def test_softmax_is_numerically_stable_on_large_inputs() -> None:
    x = np.array([1000.0, 1001.0, 1002.0])
    out = _softmax(x, axis=-1)

    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.sum(), 1.0)


def test_softmax_is_uniform_when_all_inputs_equal() -> None:
    x = np.zeros((4,))
    out = _softmax(x, axis=-1)

    np.testing.assert_allclose(out, np.full(4, 0.25))


def _make_demographic_inputs() -> tuple[np.ndarray, np.ndarray, list[str], StaticMappingProvider]:
    doc_ids = np.array([[0, 1], [0, 2], [1, 2], [0, 1]], dtype=np.int64)
    diff_scores = np.array(
        [[2.0, 0.0], [1.0, 1.0], [0.0, 5.0], [3.0, 3.0]],
        dtype=np.float64,
    )
    demographic_labels = ["A", "B", "A", "B"]
    provider = StaticMappingProvider(
        [
            [("cardio", 1.0)],
            [("neuro", 1.0)],
            [("onco", 1.0)],
        ]
    )
    return doc_ids, diff_scores, demographic_labels, provider


def test_build_keyword_demographic_table_shape_and_normalization() -> None:
    doc_ids, diff_scores, labels, provider = _make_demographic_inputs()

    table, bin_labels, keyword_labels = build_keyword_demographic_table(
        doc_ids,
        diff_scores,
        labels,
        provider,
    )

    assert table.shape == (len(bin_labels), len(keyword_labels))
    assert set(bin_labels) == {"A", "B"}
    assert set(keyword_labels) <= {"cardio", "neuro", "onco"}
    # Each bin row is divided by its patient count (2), so values are bounded by 1.
    assert table.max() <= 1.0 + 1e-9


def test_build_keyword_demographic_table_aggregates_uniform_softmax() -> None:
    doc_ids = np.array([[0, 1]], dtype=np.int64)
    diff_scores = np.array([[0.0, 0.0]], dtype=np.float64)
    labels = ["A"]
    provider = StaticMappingProvider([[("cardio", 1.0)], [("neuro", 1.0)]])

    table, bin_labels, keyword_labels = build_keyword_demographic_table(
        doc_ids,
        diff_scores,
        labels,
        provider,
    )

    assert bin_labels == ["A"]
    row = dict(zip(keyword_labels, table[0], strict=True))
    assert row["cardio"] == pytest.approx(0.5)
    assert row["neuro"] == pytest.approx(0.5)


def test_build_keyword_demographic_table_respects_explicit_bin_order() -> None:
    doc_ids, diff_scores, labels, provider = _make_demographic_inputs()

    _, bin_labels, _ = build_keyword_demographic_table(
        doc_ids,
        diff_scores,
        labels,
        provider,
        bin_order=["B", "A", "MISSING"],
    )

    assert bin_labels == ["B", "A"]


def test_build_keyword_demographic_table_trims_to_top_n_keywords() -> None:
    doc_ids, diff_scores, labels, provider = _make_demographic_inputs()

    _, _, keyword_labels = build_keyword_demographic_table(
        doc_ids,
        diff_scores,
        labels,
        provider,
        top_n_keywords=1,
    )

    assert len(keyword_labels) == 1


def test_build_keyword_demographic_table_preserves_first_seen_bin_order() -> None:
    doc_ids = np.array([[0], [0], [0]], dtype=np.int64)
    diff_scores = np.zeros((3, 1), dtype=np.float64)
    labels = ["B", "A", "B"]
    provider = StaticMappingProvider([[("only", 1.0)]])

    _, bin_labels, _ = build_keyword_demographic_table(doc_ids, diff_scores, labels, provider)

    assert bin_labels == ["B", "A"]


def test_build_keyword_demographic_table_distributes_multi_topic_weights() -> None:
    doc_ids = np.array([[0]], dtype=np.int64)
    diff_scores = np.array([[0.0]], dtype=np.float64)
    labels = ["A"]
    provider = StaticMappingProvider([[("neuro", 0.75), ("cardio", 0.25)]])

    table, _, keyword_labels = build_keyword_demographic_table(doc_ids, diff_scores, labels, provider)
    row = dict(zip(keyword_labels, table[0], strict=True))

    assert row["neuro"] == pytest.approx(0.75)
    assert row["cardio"] == pytest.approx(0.25)


def test_build_keyword_demographic_table_rejects_bad_doc_ids_shape() -> None:
    with pytest.raises(ValueError, match="doc_ids must be"):
        build_keyword_demographic_table(
            np.zeros((3,), dtype=np.int64),
            np.zeros((3, 2)),
            ["A", "B", "C"],
            StaticMappingProvider([[("x", 1.0)]]),
        )


def test_build_keyword_demographic_table_rejects_mismatched_scores() -> None:
    with pytest.raises(ValueError, match="diff_scores shape"):
        build_keyword_demographic_table(
            np.zeros((2, 2), dtype=np.int64),
            np.zeros((2, 3)),
            ["A", "B"],
            StaticMappingProvider([[("x", 1.0)]]),
        )


def test_build_keyword_demographic_table_rejects_mismatched_labels() -> None:
    with pytest.raises(ValueError, match="demographic_labels length"):
        build_keyword_demographic_table(
            np.zeros((2, 2), dtype=np.int64),
            np.zeros((2, 2)),
            ["A"],
            StaticMappingProvider([[("x", 1.0)]]),
        )


def test_build_patient_demographic_frame_computes_age_years_and_bin() -> None:
    val_schema = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "end_event_index": [10, 20],
            "prediction_time": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
        }
    )
    demographics = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "birth_time": [datetime(1970, 1, 1), datetime(2015, 1, 1)],
            "gender": ["M", "F"],
            "race": ["WHITE", "BLACK/AFRICAN AMERICAN"],
        }
    )

    result = build_patient_demographic_frame(val_schema, demographics)

    assert result.height == 2
    assert set(result.columns) >= {"subject_id", "age_years", "age_bin", "gender", "race"}
    ages = result["age_years"].to_list()
    bins = result["age_bin"].to_list()
    assert ages[0] == pytest.approx(50.0, abs=0.1)
    assert ages[1] == pytest.approx(5.0, abs=0.1)
    assert bins[0] == "45-60"
    assert bins[1] == "0-18"


def test_build_patient_demographic_frame_handles_missing_birth_time() -> None:
    val_schema = pl.DataFrame(
        {
            "subject_id": [1],
            "end_event_index": [5],
            "prediction_time": [datetime(2020, 1, 1)],
        }
    )
    demographics = pl.DataFrame(
        {
            "subject_id": pl.Series([1], dtype=pl.Int64),
            "birth_time": pl.Series([None], dtype=pl.Datetime("us")),
            "gender": pl.Series([None], dtype=pl.Utf8),
            "race": pl.Series([None], dtype=pl.Utf8),
        }
    )

    result = build_patient_demographic_frame(val_schema, demographics)

    assert result.height == 1
    # Null birth_time produces either a null age_bin or the explicit unknown label.
    assert result["age_bin"].to_list()[0] in (None, UNKNOWN_BIN)


def test_build_doc_id_to_row_map_returns_none_without_doc_ids_column(tmp_path: Path) -> None:
    dataset_path = tmp_path / "retrieval_db"
    Dataset.from_dict({"title": ["A", "B"]}).save_to_disk(str(dataset_path))

    from datasets import load_from_disk

    ds = load_from_disk(str(dataset_path))
    assert _build_doc_id_to_row_map(ds) is None


def test_build_doc_id_to_row_map_builds_lookup_for_doc_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "retrieval_db"
    Dataset.from_dict({"title": ["A", "B"], "doc_ids": [11, 22]}).save_to_disk(str(dataset_path))

    from datasets import load_from_disk

    ds = load_from_disk(str(dataset_path))
    mapping = _build_doc_id_to_row_map(ds)

    assert mapping == {11: 0, 22: 1}


def test_resolve_doc_row_index_returns_mapped_value() -> None:
    mapping = {"doc_a": 0, "doc_b": 1}
    assert _resolve_doc_row_index("doc_b", n_rows=2, doc_id_to_row=mapping) == 1


def test_resolve_doc_row_index_falls_back_to_int_when_missing_from_map() -> None:
    mapping = {"doc_a": 0}
    assert _resolve_doc_row_index(1, n_rows=5, doc_id_to_row=mapping) == 1


def test_resolve_doc_row_index_raises_key_error_on_unknown_string_id() -> None:
    mapping = {"doc_a": 0}
    with pytest.raises(KeyError, match="cannot be interpreted"):
        _resolve_doc_row_index("missing", n_rows=5, doc_id_to_row=mapping)


def test_resolve_doc_row_index_raises_key_error_on_out_of_range_fallback() -> None:
    mapping = {"doc_a": 0}
    with pytest.raises(KeyError, match="not a valid row index"):
        _resolve_doc_row_index(99, n_rows=2, doc_id_to_row=mapping)


def test_resolve_doc_row_index_without_mapping_uses_int_cast() -> None:
    assert _resolve_doc_row_index(2, n_rows=5, doc_id_to_row=None) == 2


def test_resolve_doc_row_index_without_mapping_rejects_out_of_range() -> None:
    with pytest.raises(IndexError, match="out of range"):
        _resolve_doc_row_index(99, n_rows=2, doc_id_to_row=None)


# ---------------------------------------------------------------------------
# LDATopicProvider / _fit_lda
# ---------------------------------------------------------------------------


def _build_lda_corpus(dataset_path: Path, *, content_col: str = "content") -> None:
    """Build a small corpus satisfying CountVectorizer min_df=5 (two templates, six copies each)."""
    cardio = "heart cardio blood artery pressure vessel cardiac"
    neuro = "brain nerve neuron cortex synapse cerebellum spinal"
    texts = [cardio] * 6 + [neuro] * 6
    Dataset.from_dict({content_col: texts}).save_to_disk(str(dataset_path))


def test_lda_topic_provider_returns_normalized_weighted_keywords(tmp_path: Path) -> None:
    dataset_path = tmp_path / "retrieval_db"
    _build_lda_corpus(dataset_path)

    provider = LDATopicProvider(dataset_path, n_topics=2, n_top_words=3, min_topic_weight=0.01)

    assert len(provider.vocab) == 2
    for doc_id in (0, 6):
        pairs = provider.keywords_for(doc_id)
        assert pairs
        assert sum(w for _, w in pairs) == pytest.approx(1.0)
        for label, _ in pairs:
            assert label in provider.vocab


def test_lda_topic_provider_reuses_cached_model_on_second_instantiation(tmp_path: Path) -> None:
    dataset_path = tmp_path / "retrieval_db"
    _build_lda_corpus(dataset_path)

    provider1 = LDATopicProvider(dataset_path, n_topics=2, n_top_words=3)
    cache_file = dataset_path / "lda_cache" / "lda_t2.joblib"
    assert cache_file.is_file()

    provider2 = LDATopicProvider(dataset_path, n_topics=2, n_top_words=3)
    assert provider2.vocab == provider1.vocab


def test_lda_topic_provider_falls_back_to_argmax_when_all_weights_below_threshold(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "retrieval_db"
    _build_lda_corpus(dataset_path)

    provider = LDATopicProvider(dataset_path, n_topics=2, n_top_words=3, min_topic_weight=100.0)
    pairs = provider.keywords_for(0)

    assert len(pairs) == 1
    assert pairs[0][1] == pytest.approx(1.0)


def test_lda_topic_provider_accepts_contents_column_spelling(tmp_path: Path) -> None:
    dataset_path = tmp_path / "retrieval_db"
    _build_lda_corpus(dataset_path, content_col="contents")

    provider = LDATopicProvider(dataset_path, n_topics=2, n_top_words=3)
    assert len(provider.vocab) == 2


# ---------------------------------------------------------------------------
# extract_val_schema
# ---------------------------------------------------------------------------


class _FakeValDataset:
    def __init__(self, schema_df: object) -> None:
        self.schema_df = schema_df


class _FakeDatamodule:
    def __init__(self, schema_df: object) -> None:
        self._schema_df = schema_df
        self.val_dataset: _FakeValDataset | None = None

    def setup(self, stage: str) -> None:
        assert stage == "fit"
        self.val_dataset = _FakeValDataset(self._schema_df)


def test_extract_val_schema_returns_only_the_required_columns() -> None:
    schema = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "end_event_index": [5, 10],
            "prediction_time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "extra_column": ["a", "b"],
        }
    )
    dm = _FakeDatamodule(schema)

    result = extract_val_schema(dm)

    assert result.columns == ["subject_id", "end_event_index", "prediction_time"]
    assert result.height == 2


def test_extract_val_schema_collects_lazy_frame() -> None:
    schema = pl.DataFrame(
        {
            "subject_id": [1],
            "end_event_index": [5],
            "prediction_time": [datetime(2020, 1, 1)],
        }
    ).lazy()
    dm = _FakeDatamodule(schema)

    result = extract_val_schema(dm)

    assert isinstance(result, pl.DataFrame)
    assert result.height == 1


def test_extract_val_schema_raises_on_missing_columns() -> None:
    schema = pl.DataFrame({"subject_id": [1], "end_event_index": [5]})
    dm = _FakeDatamodule(schema)

    with pytest.raises(RuntimeError, match="missing columns"):
        extract_val_schema(dm)


# ---------------------------------------------------------------------------
# load_subject_demographics
# ---------------------------------------------------------------------------


def test_load_subject_demographics_extracts_birth_gender_and_race(tmp_path: Path) -> None:
    data_dir = tmp_path / "cohort" / "data" / "train"
    data_dir.mkdir(parents=True)
    shard = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2],
            "time": [
                datetime(1980, 1, 1),
                datetime(2020, 1, 1),
                datetime(2020, 1, 1),
                datetime(1990, 6, 1),
                datetime(2020, 1, 1),
            ],
            "code": [
                "MEDS_BIRTH",
                "GENDER//M",
                "HOSPITAL_ADMISSION",
                "MEDS_BIRTH",
                "GENDER//F",
            ],
            "race": [None, None, "WHITE", None, None],
        }
    )
    shard.write_parquet(data_dir / "shard.parquet")

    result = load_subject_demographics(tmp_path / "cohort", [1, 2]).sort("subject_id")

    assert result["birth_time"].to_list() == [datetime(1980, 1, 1), datetime(1990, 6, 1)]
    assert result["gender"].to_list() == ["M", "F"]
    assert result["race"].to_list() == ["WHITE", None]


def test_load_subject_demographics_handles_missing_race_column(tmp_path: Path) -> None:
    data_dir = tmp_path / "cohort" / "data" / "train"
    data_dir.mkdir(parents=True)
    shard = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "time": [datetime(1970, 1, 1), datetime(2020, 1, 1)],
            "code": ["MEDS_BIRTH", "GENDER//M"],
        }
    )
    shard.write_parquet(data_dir / "shard.parquet")

    result = load_subject_demographics(tmp_path / "cohort", [1])

    assert result["race"].to_list() == [None]


# ---------------------------------------------------------------------------
# render_demographic_heatmaps
# ---------------------------------------------------------------------------


def test_render_demographic_heatmaps_writes_three_pngs_and_returns_tables(tmp_path: Path) -> None:
    """Writes one PNG per demographic axis to ``output_dir`` and returns per-axis tables."""
    import torch

    artifacts = {
        "doc_ids": torch.tensor([[[0, 1]], [[0, 2]], [[1, 2]]], dtype=torch.long),
        "differentiable_doc_scores": torch.tensor([[1.0, 0.5], [0.5, 1.0], [0.2, 0.8]]),
    }
    provider = StaticMappingProvider([[("a", 1.0)], [("b", 1.0)], [("c", 1.0)]])
    patient_frame = pl.DataFrame(
        {
            "age_bin": ["0-18", "18-30", "30-45"],
            "race": ["WHITE", "BLACK/AFRICAN AMERICAN", None],
            "gender": ["M", "F", None],
        }
    )

    result = render_demographic_heatmaps(artifacts, provider, patient_frame, output_dir=tmp_path)

    for axis in ("age", "race", "gender"):
        assert (tmp_path / f"keyword_demographic_{axis}.pdf").is_file(), axis
    # The legacy combined file is no longer produced.
    assert not (tmp_path / "keyword_demographic_heatmap.pdf").exists()
    # Chronic-comorbidity panel is opt-in; without ``comorbidity_frame`` it
    # should not appear.
    assert not (tmp_path / "keyword_demographic_chronic.pdf").exists()
    assert set(result.keys()) == {"age", "race", "gender"}
    for key in ("age", "race", "gender"):
        assert "table" in result[key]
        assert "bins" in result[key]
        assert "keywords" in result[key]


def test_render_demographic_heatmaps_rejects_missing_artifact_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="doc_ids"):
        render_demographic_heatmaps(
            {"unrelated": None},
            StaticMappingProvider([[("a", 1.0)]]),
            pl.DataFrame({"age_bin": ["0-18"], "race": ["WHITE"], "gender": ["M"]}),
            output_dir=tmp_path,
        )


def test_render_demographic_heatmaps_rejects_row_count_mismatch(tmp_path: Path) -> None:
    import torch

    artifacts = {
        "doc_ids": torch.tensor([[[0, 1]]], dtype=torch.long),
        "differentiable_doc_scores": torch.tensor([[1.0, 0.5]]),
    }
    patient_frame = pl.DataFrame(
        {
            "age_bin": ["0-18", "18-30"],
            "race": ["WHITE", "BLACK/AFRICAN AMERICAN"],
            "gender": ["M", "F"],
        }
    )

    with pytest.raises(RuntimeError, match="dataloader order"):
        render_demographic_heatmaps(
            artifacts,
            StaticMappingProvider([[("a", 1.0)], [("b", 1.0)]]),
            patient_frame,
            output_dir=tmp_path,
        )


def test_render_demographic_heatmaps_displays_placeholder_when_tables_are_empty(tmp_path: Path) -> None:
    """Exercises the ``if table.size == 0`` placeholder branch: each per-axis PNG is still written (with the
    placeholder text) even when no rows are available."""
    import torch

    artifacts = {
        "doc_ids": torch.zeros((0, 1, 2), dtype=torch.long),
        "differentiable_doc_scores": torch.zeros((0, 2), dtype=torch.float32),
    }
    patient_frame = pl.DataFrame(
        {
            "age_bin": pl.Series([], dtype=pl.Utf8),
            "race": pl.Series([], dtype=pl.Utf8),
            "gender": pl.Series([], dtype=pl.Utf8),
        }
    )

    result = render_demographic_heatmaps(
        artifacts,
        StaticMappingProvider([[("x", 1.0)]]),
        patient_frame,
        output_dir=tmp_path,
    )

    for axis in ("age", "race", "gender"):
        assert (tmp_path / f"keyword_demographic_{axis}.pdf").is_file(), axis
        assert result[axis]["table"].size == 0


# ---------------------------------------------------------------------------
# build_comorbidity_keyword_table + chronic heatmap panel
# ---------------------------------------------------------------------------


def test_build_comorbidity_keyword_table_multi_membership_sums_correctly() -> None:
    """A patient flagged for multiple categories contributes to every one of those rows; a patient flagged for
    none lands in the optional 'None' bucket.

    Each row is L1-normalized like the demographic version.
    """
    doc_ids = np.array([[0, 1], [0, 1], [1, 0]], dtype=np.int64)
    diff_scores = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    # Patient 0: both categories. Patient 1: cat_A only. Patient 2: neither.
    mask = np.array([[True, True], [True, False], [False, False]], dtype=bool)
    provider = StaticMappingProvider([[("a", 1.0)], [("b", 1.0)]])

    table, bin_labels, kw_labels = build_comorbidity_keyword_table(
        doc_ids,
        diff_scores,
        mask,
        category_names=["cat_A", "cat_B"],
        provider=provider,
        top_n_keywords=20,
        include_none=True,
    )

    # Row order: cat_A, cat_B, "None of the tracked".
    assert bin_labels[0] == "cat_A"
    assert bin_labels[1] == "cat_B"
    assert "None" in bin_labels[2]
    assert table.shape == (3, len(kw_labels))

    # cat_A receives contributions from patients 0 and 1 (multi-membership):
    # P0 softmax([1,0]) ≈ [0.731, 0.269] on (a, b)
    # P1 softmax([0,1]) ≈ [0.269, 0.731] on (a, b)
    # Pre-normalization sums on (a, b): (1.0, 1.0) → row normalizes to (0.5, 0.5)
    cat_a_row = table[0]
    np.testing.assert_allclose(cat_a_row, [0.5, 0.5], atol=1e-6)

    # cat_B: only P0 contributes, so the row mirrors P0's softmax.
    cat_b_row = table[1]
    np.testing.assert_allclose(cat_b_row.sum(), 1.0, atol=1e-6)
    assert cat_b_row[0] > cat_b_row[1]  # 'a' dominates

    # None row: only P2 contributes; P2's softmax([1,0]) on doc_ids[1, 0] = [1, 0]
    # → weight 0.731 on doc 1 (→ 'b') + 0.269 on doc 0 (→ 'a')
    none_row = table[2]
    np.testing.assert_allclose(none_row.sum(), 1.0, atol=1e-6)
    assert none_row[1] > none_row[0]  # 'b' dominates


def test_render_demographic_heatmaps_writes_chronic_png_when_comorbidity_frame_provided(
    tmp_path: Path,
) -> None:
    """When ``comorbidity_frame`` + ``comorbidity_categories`` are passed, the 4th PNG appears and the return
    dict gains a ``'chronic'`` key."""
    import torch

    artifacts = {
        "doc_ids": torch.tensor([[[0, 1]], [[0, 2]], [[1, 2]]], dtype=torch.long),
        "differentiable_doc_scores": torch.tensor([[1.0, 0.5], [0.5, 1.0], [0.2, 0.8]]),
    }
    provider = StaticMappingProvider([[("a", 1.0)], [("b", 1.0)], [("c", 1.0)]])
    patient_frame = pl.DataFrame(
        {
            "age_bin": ["0-18", "18-30", "30-45"],
            "race": ["WHITE", "BLACK/AFRICAN AMERICAN", None],
            "gender": ["M", "F", None],
        }
    )
    comorbidity_frame = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "Diabetes without chronic complications": [True, False, False],
            "Renal disease": [True, True, False],
        }
    )

    result = render_demographic_heatmaps(
        artifacts,
        provider,
        patient_frame,
        output_dir=tmp_path,
        comorbidity_frame=comorbidity_frame,
        comorbidity_categories=(
            "Diabetes without chronic complications",
            "Renal disease",
        ),
    )

    for axis in ("age", "race", "gender", "chronic"):
        assert (tmp_path / f"keyword_demographic_{axis}.pdf").is_file(), axis
    assert "chronic" in result
    assert "table" in result["chronic"]
    assert "bins" in result["chronic"]
    assert "keywords" in result["chronic"]


# ---------------------------------------------------------------------------
# build_pearson_residual_table + residual companion heatmap
# ---------------------------------------------------------------------------


def test_pearson_residual_table_zero_when_bin_matches_population() -> None:
    """When every bin's distribution equals the population's, all residuals should be ~0 (no over/under-
    representation)."""
    bin_to_keyword_mass = {
        "A": {"k1": 1.0, "k2": 0.5},
        "B": {"k1": 1.0, "k2": 0.5},
    }
    bin_counts = {"A": 2, "B": 2}
    pop_keyword_mass = {"k1": 2.0, "k2": 1.0}
    population_n = 4

    residuals = build_pearson_residual_table(
        bin_to_keyword_mass,
        bin_counts,
        pop_keyword_mass,
        population_n,
        bin_order=["A", "B"],
        keyword_order=["k1", "k2"],
    )

    assert residuals.shape == (2, 2)
    np.testing.assert_allclose(residuals, np.zeros((2, 2)), atol=1e-9)


def test_pearson_residual_table_positive_when_bin_over_represents_topic() -> None:
    """A bin retrieving topic k1 disproportionately gets a large positive residual at (bin, k1) and large
    negative at (bin, k2)."""
    # Two bins, 50 patients each, K=2 keywords. Bin A heavily favors k1,
    # bin B heavily favors k2. With these magnitudes the residuals should
    # comfortably exceed the |z| > 2 significance threshold.
    bin_to_keyword_mass = {
        "A": {"k1": 40.0, "k2": 10.0},
        "B": {"k1": 10.0, "k2": 40.0},
    }
    bin_counts = {"A": 50, "B": 50}
    pop_keyword_mass = {"k1": 50.0, "k2": 50.0}
    population_n = 100

    residuals = build_pearson_residual_table(
        bin_to_keyword_mass,
        bin_counts,
        pop_keyword_mass,
        population_n,
        bin_order=["A", "B"],
        keyword_order=["k1", "k2"],
    )

    # Expected mass for any cell = 50 * 50 / 100 = 25; sqrt(E) = 5
    # residual(A, k1) = (40 - 25) / 5 = +3.0; (A, k2) = (10 - 25) / 5 = -3.0
    np.testing.assert_allclose(residuals[0, 0], 3.0, atol=1e-9)
    np.testing.assert_allclose(residuals[0, 1], -3.0, atol=1e-9)
    np.testing.assert_allclose(residuals[1, 0], -3.0, atol=1e-9)
    np.testing.assert_allclose(residuals[1, 1], 3.0, atol=1e-9)
    # Cross above the conventional |z| > 2 significance threshold.
    assert np.abs(residuals).max() > 2.0


def test_pearson_residual_table_handles_empty_bin_with_nan() -> None:
    """A bin with zero patients gets an all-NaN row (no division-by-zero explosion, no crash)."""
    bin_to_keyword_mass = {
        "A": {"k1": 10.0, "k2": 5.0},
        "C": {"k1": 0.0, "k2": 0.0},  # empty bin
    }
    bin_counts = {"A": 15, "C": 0}
    pop_keyword_mass = {"k1": 10.0, "k2": 5.0}
    population_n = 15

    residuals = build_pearson_residual_table(
        bin_to_keyword_mass,
        bin_counts,
        pop_keyword_mass,
        population_n,
        bin_order=["A", "C"],
        keyword_order=["k1", "k2"],
    )

    # Row "A" finite, row "C" all-NaN.
    assert np.isfinite(residuals[0]).all()
    assert np.isnan(residuals[1]).all()


def test_pearson_residual_table_zero_for_zero_expected_cells() -> None:
    """A topic with zero population mass gives expected==0; the residual is defined as 0 there (no nonzero
    observed minus zero expected to report)."""
    bin_to_keyword_mass = {
        "A": {"k1": 5.0, "k_unused": 0.0},
    }
    bin_counts = {"A": 5}
    pop_keyword_mass = {"k1": 5.0, "k_unused": 0.0}
    population_n = 5

    residuals = build_pearson_residual_table(
        bin_to_keyword_mass,
        bin_counts,
        pop_keyword_mass,
        population_n,
        bin_order=["A"],
        keyword_order=["k1", "k_unused"],
    )

    np.testing.assert_allclose(residuals[0, 0], 0.0, atol=1e-9)
    np.testing.assert_allclose(residuals[0, 1], 0.0, atol=1e-9)


def test_render_demographic_heatmaps_writes_residuals_csv(tmp_path: Path) -> None:
    """Renderer also dumps a long-format residuals CSV so per-cell z-scores can be sorted / quoted in the
    paper without re-rendering PDFs."""
    import csv as csv_mod

    import torch

    artifacts = {
        "doc_ids": torch.tensor([[[0, 1]], [[0, 2]], [[1, 2]]], dtype=torch.long),
        "differentiable_doc_scores": torch.tensor([[1.0, 0.5], [0.5, 1.0], [0.2, 0.8]]),
    }
    provider = StaticMappingProvider([[("a", 1.0)], [("b", 1.0)], [("c", 1.0)]])
    patient_frame = pl.DataFrame(
        {
            "age_bin": ["0-18", "18-30", "30-45"],
            "race": ["WHITE", "BLACK/AFRICAN AMERICAN", None],
            "gender": ["M", "F", None],
        }
    )

    render_demographic_heatmaps(artifacts, provider, patient_frame, output_dir=tmp_path)

    csv_path = tmp_path / "keyword_demographic_residuals.csv"
    assert csv_path.is_file()
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        assert reader.fieldnames is not None
        assert {"axis", "bin", "keyword", "raw_mass", "z_score"}.issubset(reader.fieldnames)
        rows = list(reader)
    assert {r["axis"] for r in rows} == {"age", "race", "gender"}
    # Every row's raw_mass and z_score should round-trip to a finite float.
    for r in rows:
        assert np.isfinite(float(r["raw_mass"]))
        assert np.isfinite(float(r["z_score"]))


def test_render_demographic_heatmaps_residuals_csv_includes_chronic_when_provided(
    tmp_path: Path,
) -> None:
    """When ``comorbidity_frame`` is provided, the CSV also gets a ``chronic`` axis with one row per
    (category, keyword) cell."""
    import csv as csv_mod

    import torch

    artifacts = {
        "doc_ids": torch.tensor([[[0, 1]], [[0, 2]], [[1, 2]]], dtype=torch.long),
        "differentiable_doc_scores": torch.tensor([[1.0, 0.5], [0.5, 1.0], [0.2, 0.8]]),
    }
    provider = StaticMappingProvider([[("a", 1.0)], [("b", 1.0)], [("c", 1.0)]])
    patient_frame = pl.DataFrame(
        {
            "age_bin": ["0-18", "18-30", "30-45"],
            "race": ["WHITE", "BLACK/AFRICAN AMERICAN", None],
            "gender": ["M", "F", None],
        }
    )
    comorbidity_frame = pl.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "Diabetes without chronic complications": [True, False, False],
            "Renal disease": [True, True, False],
        }
    )

    render_demographic_heatmaps(
        artifacts,
        provider,
        patient_frame,
        output_dir=tmp_path,
        comorbidity_frame=comorbidity_frame,
        comorbidity_categories=("Diabetes without chronic complications", "Renal disease"),
    )

    csv_path = tmp_path / "keyword_demographic_residuals.csv"
    assert csv_path.is_file()
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)
    assert {r["axis"] for r in rows} == {"age", "race", "gender", "chronic"}


def test_render_demographic_heatmaps_writes_residual_companion_for_each_axis(
    tmp_path: Path,
) -> None:
    """Every demographic axis emitted by ``render_demographic_heatmaps`` must also have a ``_residual.png``
    sibling using a diverging colormap.

    The
    return dict's per-axis sub-dict gains a ``residual`` key.
    """
    import torch

    artifacts = {
        "doc_ids": torch.tensor([[[0, 1]], [[0, 2]], [[1, 2]]], dtype=torch.long),
        "differentiable_doc_scores": torch.tensor([[1.0, 0.5], [0.5, 1.0], [0.2, 0.8]]),
    }
    provider = StaticMappingProvider([[("a", 1.0)], [("b", 1.0)], [("c", 1.0)]])
    patient_frame = pl.DataFrame(
        {
            "age_bin": ["0-18", "18-30", "30-45"],
            "race": ["WHITE", "BLACK/AFRICAN AMERICAN", None],
            "gender": ["M", "F", None],
        }
    )

    result = render_demographic_heatmaps(
        artifacts,
        provider,
        patient_frame,
        output_dir=tmp_path,
    )

    for axis in ("age", "race", "gender"):
        assert (tmp_path / f"keyword_demographic_{axis}.pdf").is_file(), axis
        assert (tmp_path / f"keyword_demographic_{axis}_residual.pdf").is_file(), axis
        assert "residual" in result[axis], axis
        assert result[axis]["residual"].shape == result[axis]["table"].shape


# ---------------------------------------------------------------------------
# Mass-conservation invariants
#
# The keyword-demographic pipeline composes two independently-normalized steps:
# per-patient softmax over retrieved docs, and per-doc provider keyword weights.
# Total keyword mass per patient must equal 1.0, and per-bin table rows must
# equal 1.0 when the keyword axis is not truncated. These tests lock that
# invariant across both shipped providers.
# ---------------------------------------------------------------------------


@pytest.fixture(params=["title", "lda"])
def provider_inputs(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[np.ndarray, np.ndarray, list[str], object]:
    """Return (doc_ids, diff_scores, labels, provider) for each shipped provider."""
    doc_ids = np.array(
        [
            [0, 1, 2],
            [2, 3, 4],
            [0, 2, 4],
            [1, 3, 0],
        ],
        dtype=np.int64,
    )
    diff_scores = np.array(
        [
            [1.0, 0.5, -0.5],
            [2.0, 1.0, 0.0],
            [0.3, 0.7, 0.1],
            [-1.0, 0.5, 2.0],
        ],
        dtype=np.float64,
    )
    labels = ["A", "B", "A", "B"]

    if request.param == "title":
        dataset_path = tmp_path / "title_db"
        Dataset.from_dict(
            {"title": ["Cardiology", "Obstetrics", "Neurology", "Hematology", "Oncology"]}
        ).save_to_disk(str(dataset_path))
        provider = TitleKeywordProvider(dataset_path)
    else:
        dataset_path = tmp_path / "lda_db"
        _build_lda_corpus(dataset_path)
        provider = LDATopicProvider(dataset_path, n_topics=2, n_top_words=3, min_topic_weight=0.01)
        # LDA corpus has 12 docs; remap to the first 5 rows.
        doc_ids = np.array(
            [
                [0, 1, 2],
                [2, 6, 7],
                [0, 2, 8],
                [1, 6, 0],
            ],
            dtype=np.int64,
        )

    return doc_ids, diff_scores, labels, provider


def test_total_keyword_mass_per_patient_equals_one(
    provider_inputs: tuple[np.ndarray, np.ndarray, list[str], object],
) -> None:
    """Σ_k Σ_kw softmax_weight x keyword_weight = 1 for every patient."""
    doc_ids, diff_scores, _, provider = provider_inputs

    weights = _softmax(diff_scores, axis=-1)
    n_patients, k_docs = doc_ids.shape

    totals = np.zeros(n_patients)
    for i in range(n_patients):
        for k in range(k_docs):
            for _, kw_weight in provider.keywords_for(int(doc_ids[i, k])):
                totals[i] += weights[i, k] * kw_weight

    np.testing.assert_allclose(totals, 1.0, atol=1e-9)


def test_demographic_table_row_sums_equal_one_when_not_truncated(
    provider_inputs: tuple[np.ndarray, np.ndarray, list[str], object],
) -> None:
    """Per-bin rows sum to 1 when top_n_keywords covers the full active vocab."""
    doc_ids, diff_scores, labels, provider = provider_inputs

    table, _, _ = build_keyword_demographic_table(
        doc_ids,
        diff_scores,
        labels,
        provider,
        top_n_keywords=max(1000, len(provider.vocab)),
    )

    np.testing.assert_allclose(table.sum(axis=1), 1.0, atol=1e-9)


def test_demographic_table_row_sums_leq_one_when_truncated(
    provider_inputs: tuple[np.ndarray, np.ndarray, list[str], object],
) -> None:
    """top_n_keywords=1 removes mass; every row sum ≤ 1, and at least one is strictly <1."""
    doc_ids, diff_scores, labels, provider = provider_inputs

    table, _, _ = build_keyword_demographic_table(
        doc_ids,
        diff_scores,
        labels,
        provider,
        top_n_keywords=1,
    )

    row_sums = table.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-9).all()
    assert (row_sums < 1.0 - 1e-9).any(), (
        f"Expected truncation to remove mass from at least one bin; got row sums {row_sums}"
    )


def test_demographic_table_is_nonnegative_and_finite() -> None:
    """Uneven bin sizes must not produce NaN, Inf, or negative entries."""
    doc_ids = np.array(
        [[0, 1], [0, 2], [1, 2], [0, 1], [2, 0], [1, 0]],
        dtype=np.int64,
    )
    diff_scores = np.array(
        [
            [2.0, 0.0],
            [1.0, 1.0],
            [0.0, 5.0],
            [3.0, 3.0],
            [-2.0, 4.0],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    # 5 patients in "A", 1 in "B" — exercises the denom=max(count,1) path.
    labels = ["A", "A", "A", "A", "A", "B"]
    provider = StaticMappingProvider(
        [
            [("cardio", 1.0)],
            [("neuro", 0.6), ("onco", 0.4)],
            [("onco", 1.0)],
        ]
    )

    table, _, _ = build_keyword_demographic_table(doc_ids, diff_scores, labels, provider)

    assert np.isfinite(table).all()
    assert (table >= 0.0).all()


def test_mass_conservation_with_extreme_diff_scores() -> None:
    """Extreme diff_scores must not break softmax numerical stability."""
    doc_ids = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
    diff_scores = np.array(
        [
            [1e6, -1e6, 0.0],
            [-1e6, 1e6, 1e6],
        ],
        dtype=np.float64,
    )
    labels = ["A", "A"]
    provider = StaticMappingProvider(
        [
            [("cardio", 0.7), ("neuro", 0.3)],
            [("onco", 1.0)],
            [("cardio", 0.5), ("onco", 0.5)],
        ]
    )

    table, _, _ = build_keyword_demographic_table(
        doc_ids,
        diff_scores,
        labels,
        provider,
        top_n_keywords=1000,
    )

    np.testing.assert_allclose(table.sum(axis=1), 1.0, atol=1e-9)


def test_duplicate_doc_ids_preserve_mass() -> None:
    """Retrieving the same doc K times must still conserve mass."""
    doc_ids = np.array([[3, 3, 3], [2, 2, 2]], dtype=np.int64)
    diff_scores = np.array(
        [
            [1.0, 2.0, 0.5],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = ["A", "B"]
    provider = StaticMappingProvider(
        [
            [("a", 1.0)],
            [("b", 1.0)],
            [("c", 0.4), ("d", 0.6)],
            [("e", 0.2), ("f", 0.8)],
        ]
    )

    table, _, _ = build_keyword_demographic_table(
        doc_ids,
        diff_scores,
        labels,
        provider,
        top_n_keywords=1000,
    )

    np.testing.assert_allclose(table.sum(axis=1), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Coverage gap-fillers
# ---------------------------------------------------------------------------


def _provider2() -> StaticMappingProvider:
    return StaticMappingProvider([[("kw0", 1.0)], [("kw1", 1.0)]])


def test_build_keyword_demographic_table_raises_when_doc_ids_not_2d() -> None:
    """Guard at line 522 in _accumulate_demographic_bin_mass."""
    with pytest.raises(ValueError, match=r"doc_ids must be \(N, K\)"):
        build_keyword_demographic_table(
            np.zeros(5, dtype=int),
            np.zeros((5, 2)),
            ["a"] * 5,
            _provider2(),
        )


def test_build_keyword_demographic_table_raises_when_diff_scores_shape_mismatch() -> None:
    """Guard at line 524."""
    with pytest.raises(ValueError, match=r"diff_scores shape"):
        build_keyword_demographic_table(
            np.zeros((3, 2), dtype=int),
            np.zeros((3, 3)),
            ["a", "b", "a"],
            _provider2(),
        )


def test_build_keyword_demographic_table_raises_when_labels_length_mismatch() -> None:
    """Guard at line 526."""
    with pytest.raises(ValueError, match=r"demographic_labels length"):
        build_keyword_demographic_table(
            np.zeros((3, 2), dtype=int),
            np.zeros((3, 2)),
            ["a", "b"],
            _provider2(),
        )


def test_accumulate_demographic_bin_mass_raises_on_each_private_shape_guard() -> None:
    """Guards at lines 522, 524, 526 in the *private* _accumulate_demographic_bin_mass.

    These duplicate the public guards (660/662/664 in build_keyword_demographic_table)
    so the public-API tests fire the public ones first and never reach these. Cover
    them by importing the private function directly.
    """
    from medrap.demographic_analysis import _accumulate_demographic_bin_mass

    provider = _provider2()
    with pytest.raises(ValueError, match=r"doc_ids must be \(N, K\)"):
        _accumulate_demographic_bin_mass(
            np.zeros(5, dtype=int), np.zeros((5, 2)), ["a"] * 5, provider
        )
    with pytest.raises(ValueError, match=r"diff_scores shape"):
        _accumulate_demographic_bin_mass(
            np.zeros((3, 2), dtype=int), np.zeros((3, 3)), ["a", "b", "a"], provider
        )
    with pytest.raises(ValueError, match=r"demographic_labels length"):
        _accumulate_demographic_bin_mass(
            np.zeros((3, 2), dtype=int), np.zeros((3, 2)), ["a", "b"], provider
        )


def test_build_comorbidity_keyword_table_raises_on_each_public_shape_guard() -> None:
    """Guards at lines 746, 748, 751, 753, 755 in build_comorbidity_keyword_table."""
    good_doc_ids = np.zeros((3, 2), dtype=int)
    good_scores = np.zeros((3, 2))
    good_mask = np.zeros((3, 1), dtype=bool)
    good_cats = ["cat0"]

    with pytest.raises(ValueError, match=r"doc_ids must be \(N, K\)"):
        build_comorbidity_keyword_table(
            np.zeros(5, dtype=int), good_scores, good_mask, good_cats, _provider2()
        )
    with pytest.raises(ValueError, match=r"diff_scores shape"):
        build_comorbidity_keyword_table(
            good_doc_ids, np.zeros((3, 3)), good_mask, good_cats, _provider2()
        )
    with pytest.raises(ValueError, match=r"comorbidity_mask must be \(N, C\)"):
        build_comorbidity_keyword_table(
            good_doc_ids, good_scores, np.zeros(3, dtype=bool), good_cats, _provider2()
        )
    with pytest.raises(ValueError, match=r"comorbidity_mask N="):
        build_comorbidity_keyword_table(
            good_doc_ids, good_scores, np.zeros((4, 1), dtype=bool), good_cats, _provider2()
        )
    with pytest.raises(ValueError, match=r"comorbidity_mask C="):
        build_comorbidity_keyword_table(
            good_doc_ids, good_scores, np.zeros((3, 2), dtype=bool), good_cats, _provider2()
        )


def test_accumulate_comorbidity_bin_mass_raises_on_each_private_shape_guard() -> None:
    """Guards at lines 579, 581, 584, 586, 588 in _accumulate_comorbidity_bin_mass."""
    from medrap.demographic_analysis import _accumulate_comorbidity_bin_mass

    good_doc_ids = np.zeros((3, 2), dtype=int)
    good_scores = np.zeros((3, 2))
    good_mask = np.zeros((3, 1), dtype=bool)
    good_cats = ["cat0"]
    provider = _provider2()

    with pytest.raises(ValueError, match=r"doc_ids must be \(N, K\)"):
        _accumulate_comorbidity_bin_mass(
            np.zeros(5, dtype=int), good_scores, good_mask, good_cats, provider
        )
    with pytest.raises(ValueError, match=r"diff_scores shape"):
        _accumulate_comorbidity_bin_mass(
            good_doc_ids, np.zeros((3, 3)), good_mask, good_cats, provider
        )
    with pytest.raises(ValueError, match=r"comorbidity_mask must be \(N, C\)"):
        _accumulate_comorbidity_bin_mass(
            good_doc_ids, good_scores, np.zeros(3, dtype=bool), good_cats, provider
        )
    with pytest.raises(ValueError, match=r"comorbidity_mask N="):
        _accumulate_comorbidity_bin_mass(
            good_doc_ids, good_scores, np.zeros((4, 1), dtype=bool), good_cats, provider
        )
    with pytest.raises(ValueError, match=r"comorbidity_mask C="):
        _accumulate_comorbidity_bin_mass(
            good_doc_ids, good_scores, np.zeros((3, 2), dtype=bool), good_cats, provider
        )


def test_build_comorbidity_keyword_table_include_any_adds_aggregated_row() -> None:
    """Lines 780-783, 793 in build_comorbidity_keyword_table; include_any=True path."""
    doc_ids = np.array([[0, 1], [0, 1], [0, 1]])
    diff_scores = np.zeros_like(doc_ids, dtype=float)
    mask = np.array([[True, False], [False, True], [False, False]], dtype=bool)
    provider = StaticMappingProvider([[("kw0", 1.0)], [("kw1", 1.0)]])

    table, bin_labels, _ = build_comorbidity_keyword_table(
        doc_ids,
        diff_scores,
        mask,
        ["cat0", "cat1"],
        provider,
        include_any=True,
        include_none=True,
    )
    assert "Any tracked" in bin_labels
    assert "None of the tracked" in bin_labels
    any_row_idx = bin_labels.index("Any tracked")
    assert table[any_row_idx].sum() > 0.0


def test_accumulate_comorbidity_bin_mass_include_any_branch() -> None:
    """Lines 618-621 in _accumulate_comorbidity_bin_mass; include_any=True path."""
    from medrap.demographic_analysis import _accumulate_comorbidity_bin_mass

    doc_ids = np.array([[0, 1], [0, 1]])
    diff_scores = np.zeros_like(doc_ids, dtype=float)
    mask = np.array([[True, False], [False, True]], dtype=bool)
    provider = StaticMappingProvider([[("kw0", 1.0)], [("kw1", 1.0)]])
    bin_kw_mass, bin_counts, _, _ = _accumulate_comorbidity_bin_mass(
        doc_ids,
        diff_scores,
        mask,
        ["cat0", "cat1"],
        provider,
        include_any=True,
        include_none=False,
    )
    assert "Any tracked" in bin_counts
    assert bin_counts["Any tracked"] == 2
    assert "Any tracked" in bin_kw_mass


def test_render_demographic_heatmaps_rejects_comorbidity_frame_with_wrong_row_count(
    tmp_path: Path,
) -> None:
    """Guard at lines 1171-1175 in render_demographic_heatmaps."""
    import torch

    artifacts = {
        "doc_ids": torch.tensor([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=torch.long),
        "differentiable_doc_scores": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float32
        ),
    }
    patient_frame = pl.DataFrame(
        {
            "age_bin": ["18-30", "18-30", "18-30"],
            "race": ["WHITE", "WHITE", "WHITE"],
            "gender": ["M", "M", "M"],
        }
    )
    provider = StaticMappingProvider([[("kw0", 1.0)], [("kw1", 1.0)]])
    bad_comorbidity = pl.DataFrame({"cat0": [True, False]})  # 2 rows vs 3
    with pytest.raises(RuntimeError, match=r"comorbidity_frame rows"):
        render_demographic_heatmaps(
            artifacts,
            provider,
            patient_frame,
            output_dir=tmp_path,
            comorbidity_frame=bad_comorbidity,
            comorbidity_categories=["cat0"],
        )


def test_render_demographic_heatmaps_rejects_comorbidity_frame_missing_columns(tmp_path: Path) -> None:
    """Guard at line 1178."""
    import torch

    artifacts = {
        "doc_ids": torch.tensor([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=torch.long),
        "differentiable_doc_scores": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float32
        ),
    }
    patient_frame = pl.DataFrame(
        {
            "age_bin": ["18-30", "18-30", "18-30"],
            "race": ["WHITE", "WHITE", "WHITE"],
            "gender": ["M", "M", "M"],
        }
    )
    provider = StaticMappingProvider([[("kw0", 1.0)], [("kw1", 1.0)]])
    comorbidity = pl.DataFrame({"other_col": [True, True, True]})
    with pytest.raises(ValueError, match=r"missing columns for categories"):
        render_demographic_heatmaps(
            artifacts,
            provider,
            patient_frame,
            output_dir=tmp_path,
            comorbidity_frame=comorbidity,
            comorbidity_categories=["cat0"],
        )


def test_render_demographic_heatmaps_fires_significance_border_for_high_z(tmp_path: Path) -> None:
    """Covers line 1079 in render_demographic_heatmaps: |z|>2 cells get a black border patch.

    Use a maximally concentrated split: M patients retrieve doc 0 (kw_a), F patients
    retrieve doc 1 (kw_b). With N=20 (10 + 10), Pearson residual for (M, kw_a) is
    ``(10 - 5) / sqrt(5) ≈ 2.24 > 2``.
    """
    import torch

    n = 20  # N=8 gives residual sqrt(2)≈1.41 < 2; need ≥10 per bin to clear the threshold.
    doc_ids_list: list[list[list[int]]] = []
    for i in range(n):
        # First half (M): doc 0 dominant; second half (F): doc 1 dominant.
        doc_ids_list.append([[0, 1]] if i < n // 2 else [[1, 0]])
    artifacts = {
        "doc_ids": torch.tensor(doc_ids_list, dtype=torch.long),
        # Huge gap so softmax puts ~all mass on the first slot.
        "differentiable_doc_scores": torch.tensor([[50.0, -50.0]] * n, dtype=torch.float32),
    }
    patient_frame = pl.DataFrame(
        {
            "age_bin": ["18-30"] * n,
            "race": ["WHITE"] * n,
            "gender": (["M"] * (n // 2)) + (["F"] * (n // 2)),
        }
    )
    provider = StaticMappingProvider([[("kw_a", 1.0)], [("kw_b", 1.0)]])
    result = render_demographic_heatmaps(artifacts, provider, patient_frame, output_dir=tmp_path)
    gender_residual = result["gender"]["residual"]
    # At least one cell exceeds the |z|>2 significance threshold → line 1079 fires.
    assert np.nanmax(np.abs(gender_residual)) > 2.0


def test_render_demographic_heatmaps_falls_back_when_residuals_are_all_zero(tmp_path: Path) -> None:
    """Covers line 1059 in render_demographic_heatmaps: vmax==0 fallback to 1.0 when residual
    has no finite-nonzero entries.

    With every patient retrieving the same doc/keyword, observed == expected uniformly →
    residual matrix is all zero → vmax computation = 0 → fallback to 1.0.
    """
    import torch

    n = 4
    artifacts = {
        "doc_ids": torch.tensor([[[0]]] * n, dtype=torch.long),
        "differentiable_doc_scores": torch.tensor([[1.0]] * n, dtype=torch.float32),
    }
    patient_frame = pl.DataFrame(
        {
            "age_bin": ["18-30"] * n,
            "race": ["WHITE"] * n,
            "gender": ["M", "M", "F", "F"],
        }
    )
    provider = StaticMappingProvider([[("kw_uniform", 1.0)]])
    result = render_demographic_heatmaps(artifacts, provider, patient_frame, output_dir=tmp_path)
    # The gender residual should be effectively zero across all cells (no concentration).
    np.testing.assert_allclose(result["gender"]["residual"], 0.0, atol=1e-9)


def test_write_residual_csv_skips_non_finite_residual_cells(tmp_path: Path) -> None:
    """Covers line 898 in write_residual_csv: non-finite (NaN/Inf) cells are skipped in the CSV."""
    from medrap.demographic_analysis import write_residual_csv

    result = {
        "age": {
            "bins": ["18-30", "50-69"],
            "keywords": ["kw_a", "kw_b"],
            "table": np.array([[0.2, 0.3], [0.4, 0.1]]),
            # First row is all-NaN (empty bin in the underlying accumulator simulation).
            "residual": np.array([[float("nan"), float("nan")], [1.5, -0.5]]),
        }
    }
    csv_path = tmp_path / "residuals.csv"
    write_residual_csv(result, csv_path)
    text = csv_path.read_text()
    # Only the 50-69 row's cells appear in the CSV; the NaN row is silently skipped.
    assert "18-30" not in text
    assert "50-69" in text
    assert "1.500000" in text
