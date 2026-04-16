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
    build_keyword_demographic_table,
    build_patient_demographic_frame,
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


def test_render_demographic_heatmaps_writes_png_and_returns_tables(tmp_path: Path) -> None:
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
    output_path = tmp_path / "heatmap.png"

    result = render_demographic_heatmaps(artifacts, provider, patient_frame, output_path)

    assert output_path.is_file()
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
            tmp_path / "heatmap.png",
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
            tmp_path / "heatmap.png",
        )


def test_render_demographic_heatmaps_displays_placeholder_when_tables_are_empty(tmp_path: Path) -> None:
    """Exercises the ``if table.size == 0`` placeholder branch."""
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
    output_path = tmp_path / "empty.png"

    result = render_demographic_heatmaps(
        artifacts,
        StaticMappingProvider([[("x", 1.0)]]),
        patient_frame,
        output_path,
    )

    assert output_path.is_file()
    for axis in ("age", "race", "gender"):
        assert result[axis]["table"].size == 0
