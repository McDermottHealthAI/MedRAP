from __future__ import annotations

from pathlib import Path

import pytest
from datasets import Dataset

from medrap.demographic_analysis import TitleKeywordProvider


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
