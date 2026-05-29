"""Keyword x demographic aggregation for retrieval artifacts.

Given the per-patient extraction artifacts produced by
:func:`medrap.extraction.extract_artifacts`, this module aggregates which
documents (mapped to keywords) get retrieved for which kinds of patients
(binned by age, race, gender) and renders the result as a stack of heatmaps.

The doc → keyword mapping is pluggable via the :class:`DocKeywordProvider`
protocol so a future LDA-topic backend can drop in without touching the
aggregation or rendering code.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# ---------------------------------------------------------------------------
# Age bins (edit here to change all downstream binning)
# ---------------------------------------------------------------------------

AGE_BINS: list[tuple[float, float, str]] = [
    (0.0, 18.0, "0-18"),
    (18.0, 30.0, "18-30"),
    (30.0, 45.0, "30-45"),
    (45.0, 60.0, "45-60"),
    (60.0, 75.0, "60-75"),
    (75.0, float("inf"), "75+"),
]
UNKNOWN_BIN = "unknown"
AGE_BIN_ORDER: list[str] = [label for _, _, label in AGE_BINS] + [UNKNOWN_BIN]

# Race/ethnicity aggregation buckets.
RACE_AGGREGATION: dict[str, str] = {}
_RACE_BUCKETS: dict[str, list[str]] = {
    "Asian": [
        "ASIAN",
        "ASIAN - ASIAN INDIAN",
        "ASIAN - CHINESE",
        "ASIAN - KOREAN",
        "ASIAN - SOUTH EAST ASIAN",
        "ASIAN - OTHER",
    ],
    "Black": [
        "BLACK/AFRICAN AMERICAN",
        "BLACK/AFRICAN",
        "BLACK/CAPE VERDEAN",
        "BLACK/CARIBBEAN ISLAND",
        "BLACK/HAITIAN",
    ],
    "Hispanic/Latino": [
        "HISPANIC/LATINO",
        "HISPANIC OR LATINO",
        "HISPANIC/LATINO - PUERTO RICAN",
        "HISPANIC/LATINO - DOMINICAN",
        "HISPANIC/LATINO - GUATEMALAN",
        "HISPANIC/LATINO - CUBAN",
        "HISPANIC/LATINO - SALVADORAN",
        "HISPANIC/LATINO - COLOMBIAN",
        "HISPANIC/LATINO - CENTRAL AMERICAN",
        "HISPANIC/LATINO - HONDURAN",
        "HISPANIC/LATINO - MEXICAN",
        "SOUTH AMERICAN",
    ],
    "American Indian/Alaska Native": [
        "AMERICAN INDIAN/ALASKA NATIVE",
        "AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE",
    ],
    "White": [
        "WHITE",
        "WHITE - BRAZILIAN",
        "WHITE - EASTERN EUROPEAN",
        "WHITE - OTHER EUROPEAN",
        "WHITE - RUSSIAN",
        "PORTUGUESE",
    ],
    "Other/Unknown": [
        "OTHER",
        "UNKNOWN",
        "UNABLE TO OBTAIN",
        "PATIENT DECLINED TO ANSWER",
        "MULTI RACE ETHNICITY",
        "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
        "MIDDLE EASTERN",
    ],
}
for bucket, values in _RACE_BUCKETS.items():
    for v in values:
        RACE_AGGREGATION[v] = bucket
RACE_BIN_ORDER: list[str] = [
    "White",
    "Black",
    "Hispanic/Latino",
    "Asian",
    "American Indian/Alaska Native",
    "Other/Unknown",
]


def aggregate_race(raw_race: str | None) -> str:
    """Map a raw MIMIC race/ethnicity string to an aggregated bucket."""
    if raw_race is None:
        return "Other/Unknown"
    return RACE_AGGREGATION.get(raw_race.strip(), "Other/Unknown")


def bin_age(age: float | None) -> str:
    """Map a numeric age to a coarse bin label."""
    if age is None or not np.isfinite(age):
        return UNKNOWN_BIN
    for lo, hi, label in AGE_BINS:
        if lo <= age < hi:
            return label
    return UNKNOWN_BIN


# ---------------------------------------------------------------------------
# Pluggable doc → keyword mapping
# ---------------------------------------------------------------------------


class DocKeywordProvider(Protocol):
    """Maps a doc id to a list of (keyword, weight) pairs.

    Single-keyword providers (like :class:`TitleKeywordProvider`) return one
    pair with weight 1.0. Multi-topic providers (e.g. an LDA backend) can
    return several pairs whose weights sum to 1.0; the aggregator distributes
    each patient's softmax probability for that doc across the keywords in
    proportion to ``weight``.
    """

    def keywords_for(self, doc_id: int) -> list[tuple[str, float]]: ...

    @property
    def vocab(self) -> list[str]: ...


class TitleKeywordProvider:
    """One keyword per doc: the textbook chapter ``title`` field.

    Loads the HuggingFace dataset at ``retrieval_db_path`` once, materializes
    the ``title`` column, and indexes it by row id (which matches the
    ``doc_ids`` returned by retrievers configured with
    ``doc_ids_column=null``).
    """

    def __init__(self, retrieval_db_path: str | Path) -> None:
        from datasets import load_from_disk  # local import keeps module light

        ds = load_from_disk(str(retrieval_db_path))
        if "title" not in ds.column_names:
            raise ValueError(
                f"Retrieval dataset at {retrieval_db_path} has no 'title' column. "
                f"Available columns: {ds.column_names}"
            )
        self._titles: list[str] = list(ds["title"])
        self._vocab: list[str] = sorted(set(self._titles))
        self._doc_id_to_row = _build_doc_id_to_row_map(ds)

    def keywords_for(self, doc_id: int) -> list[tuple[str, float]]:
        row_idx = _resolve_doc_row_index(doc_id, n_rows=len(self._titles), doc_id_to_row=self._doc_id_to_row)
        return [(self._titles[row_idx], 1.0)]

    @property
    def vocab(self) -> list[str]:
        return self._vocab


class LDATopicProvider:
    """LDA-based multi-topic keyword provider.

    Fits sklearn ``LatentDirichletAllocation`` on the ``content`` column of the
    retrieval dataset. Each doc maps to a distribution over topics; each topic
    is labeled by its top words. The aggregator distributes a patient's softmax
    weight on each retrieved doc across that doc's topics.

    The fitted model is cached to ``{retrieval_db_path}/lda_cache/`` so
    subsequent runs skip the (slow) fitting step.

    Args:
        retrieval_db_path: HF dataset directory.
        n_topics: Number of LDA topics. 30 is a reasonable default for ~18
            medical textbooks (roughly 1-2 topics per specialty).
        n_top_words: Number of top words used to label each topic.
        min_topic_weight: Per-doc topic weights below this are dropped (avoids
            noise from near-zero loadings).
    """

    def __init__(
        self,
        retrieval_db_path: str | Path,
        *,
        n_topics: int = 30,
        n_top_words: int = 5,
        min_topic_weight: float = 0.01,
    ) -> None:
        import joblib

        cache_dir = Path(retrieval_db_path) / "lda_cache"
        cache_file = cache_dir / f"lda_t{n_topics}.joblib"

        if cache_file.is_file():
            print(f"Loading cached LDA model from {cache_file}", flush=True)
            cached = joblib.load(cache_file)
            self._doc_topic_dist = cached["doc_topic_dist"]
            self._topic_labels = cached["topic_labels"]
        else:
            print(f"Fitting LDA with {n_topics} topics...", flush=True)
            from datasets import load_from_disk

            print("  Loading retrieval corpus...", flush=True)
            ds = load_from_disk(str(retrieval_db_path))
            content_col = "content" if "content" in ds.column_names else "contents"
            texts = ds[content_col]
            print(f"  Loaded {len(texts)} documents. Vectorizing...", flush=True)

            doc_topic_dist, topic_labels = _fit_lda(texts, n_topics=n_topics, n_top_words=n_top_words)
            self._doc_topic_dist = doc_topic_dist
            self._topic_labels = topic_labels

            cache_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump({"doc_topic_dist": doc_topic_dist, "topic_labels": topic_labels}, cache_file)
            print(f"Cached LDA model to {cache_file}", flush=True)

        self._min_weight = min_topic_weight
        self._vocab = list(self._topic_labels)
        from datasets import load_from_disk

        ds = load_from_disk(str(retrieval_db_path))
        self._doc_id_to_row = _build_doc_id_to_row_map(ds)
        print(f"LDA topics: {self._topic_labels}")

    def keywords_for(self, doc_id: int) -> list[tuple[str, float]]:
        row_idx = _resolve_doc_row_index(
            doc_id, n_rows=self._doc_topic_dist.shape[0], doc_id_to_row=self._doc_id_to_row
        )
        weights = self._doc_topic_dist[row_idx]  # (n_topics,)
        pairs = [(self._topic_labels[t], float(w)) for t, w in enumerate(weights) if w >= self._min_weight]
        if not pairs:
            best = int(weights.argmax())
            pairs = [(self._topic_labels[best], float(weights[best]))]
        total = sum(w for _, w in pairs)
        return [(kw, w / total) for kw, w in pairs]

    @property
    def vocab(self) -> list[str]:
        return self._vocab


def _fit_lda(
    texts: list[str],
    *,
    n_topics: int,
    n_top_words: int,
) -> tuple[np.ndarray, list[str]]:
    """Fit LDA and return ``(doc_topic_dist, topic_labels)``.

    Uses TF (raw term counts) with English stop words removed and medical text-friendly tokenization.
    """
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=5,
        stop_words="english",
        max_features=5_000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # alpha-only, 2+ chars
    )
    tf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        learning_method="online",
        batch_size=2048,
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )
    doc_topic_dist = lda.fit_transform(tf_matrix)  # (n_docs, n_topics)

    topic_labels: list[str] = []
    for _topic_idx, topic_weights in enumerate(lda.components_):
        top_word_indices = topic_weights.argsort()[: -n_top_words - 1 : -1]
        top_words = [feature_names[i] for i in top_word_indices]
        topic_labels.append(" / ".join(top_words))

    return doc_topic_dist, topic_labels


def _build_doc_id_to_row_map(dataset) -> dict[Any, int] | None:
    """Return ``doc_id`` -> dataset row index mapping when available.

    When the retrieval dataset includes a ``doc_ids`` column, retrievers may
    emit those IDs instead of raw row indices. This helper builds an explicit
    lookup so downstream analysis can map retrieved IDs back to corpus rows.
    If ``doc_ids`` is absent, callers should treat incoming IDs as row indices.
    """
    if "doc_ids" not in dataset.column_names:
        return None

    raw_doc_ids = dataset["doc_ids"]
    mapping: dict[Any, int] = {}
    for row_idx, raw_doc_id in enumerate(raw_doc_ids):
        doc_id = raw_doc_id
        if doc_id in mapping:
            first = mapping[doc_id]
            raise ValueError(
                "Retrieval dataset has duplicate doc_ids values; "
                f"doc_id={doc_id} appears at rows {first} and {row_idx}."
            )
        mapping[doc_id] = row_idx
    return mapping


def _resolve_doc_row_index(doc_id: Any, *, n_rows: int, doc_id_to_row: dict[Any, int] | None) -> int:
    """Resolve retriever-emitted ``doc_id`` to the corpus row index."""
    if doc_id_to_row is not None:
        if doc_id in doc_id_to_row:
            return doc_id_to_row[doc_id]
        try:
            normalized_doc_id = int(doc_id)
        except (TypeError, ValueError) as exc:
            raise KeyError(
                f"Retrieved doc_id {doc_id!r} not found in retrieval dataset doc_ids "
                f"column and cannot be interpreted as a row index."
            ) from exc
        if 0 <= normalized_doc_id < n_rows:
            return normalized_doc_id
        raise KeyError(
            f"Retrieved doc_id {doc_id!r} not found in retrieval dataset doc_ids "
            f"column and is not a valid row index for {n_rows} rows."
        )

    normalized_doc_id = int(doc_id)
    if not (0 <= normalized_doc_id < n_rows):
        raise IndexError(f"Retrieved doc_id {normalized_doc_id} is out of range for {n_rows} corpus rows.")
    return normalized_doc_id


class StaticMappingProvider:
    """In-memory provider for tests."""

    def __init__(self, doc_id_to_keywords: list[list[tuple[str, float]]]) -> None:
        self._mapping = doc_id_to_keywords
        seen: set[str] = set()
        for entries in doc_id_to_keywords:
            for kw, _ in entries:
                seen.add(kw)
        self._vocab = sorted(seen)

    def keywords_for(self, doc_id: int) -> list[tuple[str, float]]:
        return list(self._mapping[doc_id])

    @property
    def vocab(self) -> list[str]:
        return self._vocab


# ---------------------------------------------------------------------------
# Subject id and demographics
# ---------------------------------------------------------------------------


def extract_val_schema(datamodule) -> pl.DataFrame:
    """Return the val split's ``(subject_id, end_event_index, prediction_time)``.

    Order matches the val dataloader (which uses ``shuffle=False``) and
    therefore the row order of the saved extraction artifacts.
    """
    datamodule.setup("fit")
    dataset = datamodule.val_dataset
    schema = dataset.schema_df
    if not isinstance(schema, pl.DataFrame):
        schema = schema.collect()
    needed = {"subject_id", "end_event_index", "prediction_time"}
    missing = needed - set(schema.columns)
    if missing:
        raise RuntimeError(f"Validation schema_df is missing columns {sorted(missing)}; got {schema.columns}")
    return schema.select(["subject_id", "end_event_index", "prediction_time"])


def load_subject_demographics(
    meds_cohort_dir: str | Path,
    subject_ids: Iterable[int],
) -> pl.DataFrame:
    """Pull birth time, gender, race/ethnicity for the requested subjects.

    Scans every parquet under ``{meds_cohort_dir}/data/{train,tuning,held_out}``
    with polars in lazy mode, filters to the requested subject_ids, and
    extracts:

    - ``MEDS_BIRTH`` events → ``birth_time`` (the event's ``time`` column)
    - ``GENDER//`` prefixed codes → ``gender`` (suffix after ``//``)
    - ``HOSPITAL_ADMISSION`` events with a ``race`` column → ``race``
      (race/ethnicity is stored as a supplementary column on admission events
      in the MIMIC-IV MEDS ETL, not as a standalone code row)

    Returns one row per subject_id with columns
    ``[subject_id, birth_time, gender, race]``. Missing values stay null.
    """
    cohort_root = Path(meds_cohort_dir)
    parquet_glob = str(cohort_root / "data" / "*" / "*.parquet")
    subject_set = list({int(s) for s in subject_ids})

    # --- Birth and gender: extracted from code column ---
    lf_code = (
        pl.scan_parquet(parquet_glob)
        .filter(pl.col("subject_id").is_in(subject_set))
        .select(["subject_id", "time", "code"])
    )

    is_birth = pl.col("code") == "MEDS_BIRTH"
    is_gender = pl.col("code").str.starts_with("GENDER//")

    relevant = lf_code.filter(is_birth | is_gender).collect()

    births = (
        relevant.filter(pl.col("code") == "MEDS_BIRTH")
        .group_by("subject_id")
        .agg(pl.col("time").min().alias("birth_time"))
    )
    genders = (
        relevant.filter(pl.col("code").str.starts_with("GENDER//"))
        .with_columns(pl.col("code").str.replace("GENDER//", "", literal=True).alias("gender"))
        .group_by("subject_id")
        .agg(pl.col("gender").first())
    )

    # --- Race/ethnicity: supplementary column on HOSPITAL_ADMISSION events ---
    # The "race" column only exists in parquets that contain admission events.
    # We try to read it; if the column doesn't exist in a shard, we skip it.
    try:
        lf_race = (
            pl.scan_parquet(parquet_glob)
            .filter(
                pl.col("subject_id").is_in(subject_set)
                & pl.col("code").str.starts_with("HOSPITAL_ADMISSION")
                & pl.col("race").is_not_null()
            )
            .select(["subject_id", "race"])
        )
        races = lf_race.collect().group_by("subject_id").agg(pl.col("race").first())
    except Exception:
        # "race" column doesn't exist in these parquets
        races = pl.DataFrame(
            {"subject_id": pl.Series([], dtype=pl.Int64), "race": pl.Series([], dtype=pl.Utf8)}
        )

    base = pl.DataFrame({"subject_id": subject_set})
    return (
        base.join(births, on="subject_id", how="left")
        .join(genders, on="subject_id", how="left")
        .join(races, on="subject_id", how="left")
    )


def build_patient_demographic_frame(
    val_schema: pl.DataFrame,
    demographics: pl.DataFrame,
) -> pl.DataFrame:
    """Join per-row val schema with per-subject demographics, compute age bin.

    The output has one row per validation sample (in dataloader order) with
    columns: ``subject_id, prediction_time, birth_time, gender, race,
    age_years, age_bin``.
    """
    joined = val_schema.join(demographics, on="subject_id", how="left")
    age_expr = (
        (pl.col("prediction_time").cast(pl.Datetime("us")) - pl.col("birth_time").cast(pl.Datetime("us")))
        .dt.total_days()
        .cast(pl.Float64)
        / 365.25
    ).alias("age_years")
    joined = joined.with_columns(age_expr)
    age_bin = pl.col("age_years").map_elements(bin_age, return_dtype=pl.Utf8).alias("age_bin")
    return joined.with_columns(age_bin)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def _accumulate_demographic_bin_mass(
    doc_ids: np.ndarray,
    diff_scores: np.ndarray,
    demographic_labels: Sequence[str],
    provider: DocKeywordProvider,
) -> tuple[
    dict[str, defaultdict[str, float]],
    defaultdict[str, int],
    defaultdict[str, float],
    int,
]:
    """Accumulate per-bin and per-population keyword mass (mutually exclusive).

    Each patient contributes exclusively to one bin (one demographic
    label). The returned ``pop_keyword_mass[t]`` equals
    ``sum_bins bin_to_keyword_mass[b][t]`` by construction.

    Returns:
        ``(bin_to_keyword_mass, bin_counts, pop_keyword_mass, N)``.
    """
    if doc_ids.ndim != 2:
        raise ValueError(f"doc_ids must be (N, K); got {doc_ids.shape}")
    if diff_scores.shape != doc_ids.shape:
        raise ValueError(f"diff_scores shape {diff_scores.shape} must match doc_ids {doc_ids.shape}")
    if len(demographic_labels) != doc_ids.shape[0]:
        raise ValueError(f"demographic_labels length {len(demographic_labels)} != N={doc_ids.shape[0]}")

    weights = _softmax(diff_scores, axis=-1)

    bin_to_keyword_mass: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    bin_counts: defaultdict[str, int] = defaultdict(int)
    pop_keyword_mass: defaultdict[str, float] = defaultdict(float)

    n, k_docs = doc_ids.shape
    for i in range(n):
        bin_label = demographic_labels[i]
        bin_counts[bin_label] += 1
        for k in range(k_docs):
            d = int(doc_ids[i, k])
            pw = float(weights[i, k])
            for kw, kw_weight in provider.keywords_for(d):
                contribution = pw * kw_weight
                bin_to_keyword_mass[bin_label][kw] += contribution
                pop_keyword_mass[kw] += contribution

    return bin_to_keyword_mass, bin_counts, pop_keyword_mass, n


def _accumulate_comorbidity_bin_mass(
    doc_ids: np.ndarray,
    diff_scores: np.ndarray,
    comorbidity_mask: np.ndarray,
    category_names: Sequence[str],
    provider: DocKeywordProvider,
    *,
    include_none: bool = True,
    include_any: bool = False,
    none_label: str = "None of the tracked",
    any_label: str = "Any tracked",
) -> tuple[
    dict[str, defaultdict[str, float]],
    defaultdict[str, int],
    defaultdict[str, float],
    int,
]:
    """Accumulate per-bin and per-population keyword mass (multi-membership).

    A patient flagged for multiple categories contributes to every category
    they're in. Critically, ``pop_keyword_mass`` is computed by counting each
    patient exactly **once** — not summing across the bin accumulator (which
    would double-count multi-flagged patients). This makes the
    ``pop_keyword_mass`` interpretable as the per-patient population
    distribution required by chi-square residual computation.

    Returns:
        ``(bin_to_keyword_mass, bin_counts, pop_keyword_mass, N)``.
    """
    if doc_ids.ndim != 2:
        raise ValueError(f"doc_ids must be (N, K); got {doc_ids.shape}")
    if diff_scores.shape != doc_ids.shape:
        raise ValueError(f"diff_scores shape {diff_scores.shape} must match doc_ids {doc_ids.shape}")
    mask = np.asarray(comorbidity_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"comorbidity_mask must be (N, C); got {mask.shape}")
    if mask.shape[0] != doc_ids.shape[0]:
        raise ValueError(f"comorbidity_mask N={mask.shape[0]} != doc_ids N={doc_ids.shape[0]}")
    if mask.shape[1] != len(category_names):
        raise ValueError(f"comorbidity_mask C={mask.shape[1]} != len(category_names)={len(category_names)}")

    weights = _softmax(diff_scores, axis=-1)

    bin_to_keyword_mass: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    bin_counts: defaultdict[str, int] = defaultdict(int)
    pop_keyword_mass: defaultdict[str, float] = defaultdict(float)

    n, k_docs = doc_ids.shape
    for i in range(n):
        # Per-patient keyword contribution computed once.
        contributions: defaultdict[str, float] = defaultdict(float)
        for k in range(k_docs):
            d = int(doc_ids[i, k])
            pw = float(weights[i, k])
            for kw, kw_weight in provider.keywords_for(d):
                contributions[kw] += pw * kw_weight

        # Population total counts each patient ONCE regardless of category flags.
        for kw, m in contributions.items():
            pop_keyword_mass[kw] += m

        any_flagged = False
        for c, cat in enumerate(category_names):
            if mask[i, c]:
                any_flagged = True
                bin_counts[cat] += 1
                for kw, m in contributions.items():
                    bin_to_keyword_mass[cat][kw] += m

        if include_any and any_flagged:
            bin_counts[any_label] += 1
            for kw, m in contributions.items():
                bin_to_keyword_mass[any_label][kw] += m
        if include_none and not any_flagged:
            bin_counts[none_label] += 1
            for kw, m in contributions.items():
                bin_to_keyword_mass[none_label][kw] += m

    return bin_to_keyword_mass, bin_counts, pop_keyword_mass, n


def build_keyword_demographic_table(
    doc_ids: np.ndarray,
    diff_scores: np.ndarray,
    demographic_labels: Sequence[str],
    provider: DocKeywordProvider,
    *,
    bin_order: Sequence[str] | None = None,
    top_n_keywords: int | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Aggregate softmax-weighted keyword mass per demographic bin.

    Args:
        doc_ids: Retrieved doc ids, shape ``(N, K)`` (after squeezing the
            ``R=1`` retrieval-query axis).
        diff_scores: Differentiable retrieval scores, shape ``(N, K)``.
        demographic_labels: Bin label per patient, length ``N``.
        provider: A :class:`DocKeywordProvider`.
        bin_order: Optional explicit ordering for the y-axis. If omitted, bins
            appear in first-seen order.
        top_n_keywords: Cap the keyword axis to the heaviest ``top_n``
            keywords (by total mass across all bins). ``None`` (default)
            or any non-positive value means "show all keywords"; useful
            so a low-mass-but-highly-distinctive keyword isn't hidden.

    Returns:
        ``(table, bin_labels, keyword_labels)`` where ``table`` has shape
        ``(len(bin_labels), len(keyword_labels))`` and each row is normalized
        by the patient count in that bin.
    """
    if doc_ids.ndim != 2:
        raise ValueError(f"doc_ids must be (N, K); got {doc_ids.shape}")
    if diff_scores.shape != doc_ids.shape:
        raise ValueError(f"diff_scores shape {diff_scores.shape} must match doc_ids {doc_ids.shape}")
    if len(demographic_labels) != doc_ids.shape[0]:
        raise ValueError(f"demographic_labels length {len(demographic_labels)} != N={doc_ids.shape[0]}")

    weights = _softmax(diff_scores, axis=-1)  # (N, K)

    # bin -> {keyword: total mass}
    bin_to_keyword_mass: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    bin_counts: defaultdict[str, int] = defaultdict(int)

    for i in range(doc_ids.shape[0]):
        bin_label = demographic_labels[i]
        bin_counts[bin_label] += 1
        for k in range(doc_ids.shape[1]):
            d = int(doc_ids[i, k])
            patient_weight = float(weights[i, k])
            for keyword, kw_weight in provider.keywords_for(d):
                bin_to_keyword_mass[bin_label][keyword] += patient_weight * kw_weight

    # Determine bin axis ordering.
    if bin_order is not None:
        bin_labels = [b for b in bin_order if b in bin_counts]
    else:
        seen: list[str] = []
        for b in demographic_labels:
            if b not in seen:
                seen.append(b)
        bin_labels = seen

    # Pick top-N keywords by total mass across all bins.
    keyword_total: dict[str, float] = defaultdict(float)
    for masses in bin_to_keyword_mass.values():
        for kw, m in masses.items():
            keyword_total[kw] += m
    ordered = sorted(keyword_total, key=lambda kw: -keyword_total[kw])
    top_keywords = ordered if top_n_keywords is None or top_n_keywords <= 0 else ordered[:top_n_keywords]

    table = np.zeros((len(bin_labels), len(top_keywords)), dtype=np.float64)
    for i, b in enumerate(bin_labels):
        denom = max(bin_counts[b], 1)
        masses = bin_to_keyword_mass[b]
        for j, kw in enumerate(top_keywords):
            table[i, j] = masses.get(kw, 0.0) / denom

    return table, bin_labels, top_keywords


def build_comorbidity_keyword_table(
    doc_ids: np.ndarray,
    diff_scores: np.ndarray,
    comorbidity_mask: np.ndarray,
    category_names: Sequence[str],
    provider: DocKeywordProvider,
    *,
    top_n_keywords: int | None = None,
    include_none: bool = True,
    include_any: bool = False,
    none_label: str = "None of the tracked",
    any_label: str = "Any tracked",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Multi-membership version of :func:`build_keyword_demographic_table`.

    Each patient's softmax-weighted keyword mass is added to *every*
    category row they are flagged for in ``comorbidity_mask``. Optional
    "None of the tracked" and "Any tracked" buckets aggregate the patients
    with no flags / at least one flag respectively. Per-row normalization
    matches the demographic table: each cell is the average per-patient
    mass for that category-keyword combination.

    Args:
        doc_ids: ``(N, K)`` retrieved doc ids.
        diff_scores: ``(N, K)`` differentiable retrieval scores.
        comorbidity_mask: ``(N, C)`` bool. Each row is the patient's
            indicator vector across ``category_names``.
        category_names: ``length C``, the canonical row order for the heatmap.
        provider: A :class:`DocKeywordProvider`.
        top_n_keywords: Keep the top-K keywords across all bins. ``None``
            (default) means "show all keywords".
        include_none: Add a "None of the tracked" row aggregating
            patients with zero flags.
        include_any: Add an "Any tracked" row aggregating patients with at
            least one flag.
    """
    if doc_ids.ndim != 2:
        raise ValueError(f"doc_ids must be (N, K); got {doc_ids.shape}")
    if diff_scores.shape != doc_ids.shape:
        raise ValueError(f"diff_scores shape {diff_scores.shape} must match doc_ids {doc_ids.shape}")
    mask = np.asarray(comorbidity_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"comorbidity_mask must be (N, C); got {mask.shape}")
    if mask.shape[0] != doc_ids.shape[0]:
        raise ValueError(f"comorbidity_mask N={mask.shape[0]} != doc_ids N={doc_ids.shape[0]}")
    if mask.shape[1] != len(category_names):
        raise ValueError(f"comorbidity_mask C={mask.shape[1]} != len(category_names)={len(category_names)}")

    weights = _softmax(diff_scores, axis=-1)  # (N, K)

    bin_to_keyword_mass: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    bin_counts: defaultdict[str, int] = defaultdict(int)

    n, k_docs = doc_ids.shape
    for i in range(n):
        # One-pass per-patient contribution dict (avoid recomputing per category).
        contributions: defaultdict[str, float] = defaultdict(float)
        for k in range(k_docs):
            d = int(doc_ids[i, k])
            pw = float(weights[i, k])
            for keyword, kw_weight in provider.keywords_for(d):
                contributions[keyword] += pw * kw_weight

        any_flagged = False
        for c, cat in enumerate(category_names):
            if mask[i, c]:
                any_flagged = True
                bin_counts[cat] += 1
                for keyword, m in contributions.items():
                    bin_to_keyword_mass[cat][keyword] += m

        if include_any and any_flagged:
            bin_counts[any_label] += 1
            for keyword, m in contributions.items():
                bin_to_keyword_mass[any_label][keyword] += m
        if include_none and not any_flagged:
            bin_counts[none_label] += 1
            for keyword, m in contributions.items():
                bin_to_keyword_mass[none_label][keyword] += m

    # Row order: categories in given order (only those with ≥1 flagged patient),
    # then "Any", then "None".
    bin_labels = [c for c in category_names if c in bin_counts]
    if include_any and any_label in bin_counts:
        bin_labels.append(any_label)
    if include_none and none_label in bin_counts:
        bin_labels.append(none_label)

    keyword_total: dict[str, float] = defaultdict(float)
    for masses in bin_to_keyword_mass.values():
        for keyword, m in masses.items():
            keyword_total[keyword] += m
    ordered = sorted(keyword_total, key=lambda kw: -keyword_total[kw])
    top_keywords = ordered if top_n_keywords is None or top_n_keywords <= 0 else ordered[:top_n_keywords]

    table = np.zeros((len(bin_labels), len(top_keywords)), dtype=np.float64)
    for i, b in enumerate(bin_labels):
        denom = max(bin_counts[b], 1)
        masses = bin_to_keyword_mass[b]
        for j, kw in enumerate(top_keywords):
            table[i, j] = masses.get(kw, 0.0) / denom

    return table, bin_labels, top_keywords


def build_pearson_residual_table(
    bin_to_keyword_mass: dict[str, dict[str, float]],
    bin_counts: dict[str, int],
    pop_keyword_mass: dict[str, float],
    population_n: int,
    bin_order: Sequence[str],
    keyword_order: Sequence[str],
) -> np.ndarray:
    """Standardized Pearson cell residuals on a (bin, keyword) contingency.

    For each cell:

        O[b, t] = bin_to_keyword_mass[b][t]
        E[b, t] = bin_counts[b] * pop_keyword_mass[t] / population_n
        residual = (O - E) / sqrt(E)

    This is the decomposition of the chi-square statistic. ``|residual| > 2``
    is the canonical ~95% one-tailed significance threshold for an
    individual cell's contribution (Agresti, *Categorical Data Analysis*,
    §3.2).

    Empty bins (``bin_counts[b] == 0``) yield an all-NaN row; cells with
    zero expected mass (a topic that nobody retrieves) yield 0.0.

    Args:
        bin_to_keyword_mass: ``{bin: {keyword: total mass}}`` accumulator.
        bin_counts: Patient count per bin.
        pop_keyword_mass: Total population mass per keyword. For
            multi-membership accumulators (e.g. Charlson) this must count
            each patient exactly once.
        population_n: Total number of patients in the population.
        bin_order: Row order for the output array.
        keyword_order: Column order for the output array.
    """
    n_bins = len(bin_order)
    n_keywords = len(keyword_order)
    residual = np.zeros((n_bins, n_keywords), dtype=np.float64)
    if population_n <= 0:
        residual[:] = np.nan
        return residual

    for i, b in enumerate(bin_order):
        n_b = bin_counts.get(b, 0)
        if n_b == 0:
            residual[i, :] = np.nan
            continue
        masses = bin_to_keyword_mass.get(b, {})
        for j, kw in enumerate(keyword_order):
            observed = float(masses.get(kw, 0.0))
            expected = n_b * float(pop_keyword_mass.get(kw, 0.0)) / population_n
            if expected > 0.0:
                residual[i, j] = (observed - expected) / np.sqrt(expected)
            else:
                residual[i, j] = 0.0
    return residual


def write_residual_csv(result: dict, output_path: Path) -> None:
    """Dump per-cell raw mass + Pearson residual in long format to a CSV.

    Columns: ``axis, bin, keyword, raw_mass, z_score``. One row per
    (axis, bin, keyword) cell with a finite residual; NaN cells (empty
    bins) are skipped. Lets the user sort / filter / cite specific
    z-scores in a paper without reopening the PDFs.

    Args:
        result: The dict returned by :func:`render_demographic_heatmaps`.
            Each value must have ``bins``, ``keywords``, ``table``
            (raw mass), and ``residual`` (z-score) keys.
        output_path: Destination CSV path. Parent directory is created.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["axis", "bin", "keyword", "raw_mass", "z_score"])
        for axis, entry in result.items():
            bins = entry["bins"]
            keywords = entry["keywords"]
            raw = entry["table"]
            residual = entry["residual"]
            for i, b in enumerate(bins):
                for j, kw in enumerate(keywords):
                    z = float(residual[i, j])
                    if not np.isfinite(z):
                        continue
                    raw_val = float(raw[i, j])
                    writer.writerow([axis, b, kw, f"{raw_val:.6f}", f"{z:.6f}"])
    print(f"Residuals CSV saved to {output_path}")


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------


def render_demographic_heatmaps(
    artifacts: dict,
    provider: DocKeywordProvider,
    patient_frame: pl.DataFrame,
    output_dir: Path,
    *,
    top_n_keywords: int | None = None,
    comorbidity_frame: pl.DataFrame | None = None,
    comorbidity_categories: Sequence[str] | None = None,
) -> dict:
    """Render demographic and optional chronic-comorbidity heatmaps.

    Each axis is written to its own PNG so the panels can be sized and
    embedded independently in paper figures::

        <output_dir>/keyword_demographic_age.pdf
        <output_dir>/keyword_demographic_age_residual.pdf
        <output_dir>/keyword_demographic_race.pdf
        <output_dir>/keyword_demographic_race_residual.pdf
        <output_dir>/keyword_demographic_gender.pdf
        <output_dir>/keyword_demographic_gender_residual.pdf
        <output_dir>/keyword_demographic_chronic.pdf            # only when comorbidity_frame is provided
        <output_dir>/keyword_demographic_chronic_residual.pdf   # only when comorbidity_frame is provided

    Args:
        artifacts: Dict loaded from ``extraction_artifacts.pt``. Must contain
            ``doc_ids`` and ``differentiable_doc_scores``.
        provider: Doc → keyword mapping.
        patient_frame: One row per validation sample, in dataloader order.
            Must contain ``age_bin``, ``race``, ``gender`` columns.
        output_dir: Directory to write the PNGs into. Created if missing.
        top_n_keywords: Per-axis cap on keyword count. ``None`` (default)
            means "show all keywords" — recommended so low-mass but
            statistically distinctive topics aren't truncated out of the
            residual heatmap.
        comorbidity_frame: Optional per-patient comorbidity flag frame. Must
            have one boolean column per name in ``comorbidity_categories``
            and row count equal to ``patient_frame``. When provided, an
            extra panel ``keyword_demographic_chronic.pdf`` is written
            with multi-membership rows.
        comorbidity_categories: Ordered list of category names matching
            ``comorbidity_frame`` columns. Required when ``comorbidity_frame``
            is provided.

    Returns:
        Dict with table arrays and labels for each axis, for downstream
        inspection / diagnostics. Has keys ``"age"``, ``"race"``, ``"gender"``,
        and (when ``comorbidity_frame`` is provided) ``"chronic"``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "doc_ids" not in artifacts or "differentiable_doc_scores" not in artifacts:
        raise ValueError(
            "artifacts must contain doc_ids and differentiable_doc_scores; "
            f"got keys {sorted(artifacts.keys())}"
        )

    doc_ids = artifacts["doc_ids"].numpy()
    if doc_ids.ndim == 3 and doc_ids.shape[1] == 1:
        doc_ids = doc_ids[:, 0, :]  # (N, K)
    diff_scores = artifacts["differentiable_doc_scores"].numpy()  # (N, K)

    if doc_ids.shape[0] != patient_frame.height:
        raise RuntimeError(
            f"Artifact rows ({doc_ids.shape[0]}) != val schema rows "
            f"({patient_frame.height}). The dataloader order assumption is "
            f"broken. Fall back to capturing subject_id inside predict_step."
        )

    age_labels = patient_frame["age_bin"].to_list()
    race_labels = [aggregate_race(r) for r in patient_frame["race"].to_list()]
    gender_labels = [g if g is not None else "unknown" for g in patient_frame["gender"].to_list()]

    output_dir.mkdir(parents=True, exist_ok=True)

    def _pick_bins_and_keywords(
        bin_to_kw_mass: dict[str, dict[str, float]],
        bin_counts: dict[str, int],
        bin_order_hint: Sequence[str] | None,
    ) -> tuple[list[str], list[str]]:
        """Apply the same row ordering + top-N keyword selection used by ``build_keyword_demographic_table``
        so raw and residual heatmaps share their axes."""
        if bin_order_hint is not None:
            bin_labels = [b for b in bin_order_hint if b in bin_counts]
        else:  # pragma: no cover - every caller in render_demographic_heatmaps passes a non-None hint
            bin_labels = list(bin_counts.keys())
        keyword_total: dict[str, float] = defaultdict(float)
        for masses in bin_to_kw_mass.values():
            for kw, m in masses.items():
                keyword_total[kw] += m
        ordered = sorted(keyword_total, key=lambda kw: -keyword_total[kw])
        top_keywords = ordered if top_n_keywords is None or top_n_keywords <= 0 else ordered[:top_n_keywords]
        return bin_labels, top_keywords

    def _normalize_table(
        bin_to_kw_mass: dict[str, dict[str, float]],
        bin_counts: dict[str, int],
        bin_labels: Sequence[str],
        top_keywords: Sequence[str],
    ) -> np.ndarray:
        """Per-patient-average normalization (the raw heatmap content)."""
        table = np.zeros((len(bin_labels), len(top_keywords)), dtype=np.float64)
        for i, b in enumerate(bin_labels):
            denom = max(bin_counts.get(b, 0), 1)
            masses = bin_to_kw_mass.get(b, {})
            for j, kw in enumerate(top_keywords):
                table[i, j] = masses.get(kw, 0.0) / denom
        return table

    def _render_one(
        table: np.ndarray,
        bins: Sequence[str],
        kws: Sequence[str],
        title: str,
        out_path: Path,
        *,
        figsize: tuple[float, float] | None = None,
        diverging: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize or (10.0, 5.0))
        no_data = (
            table.size == 0
            or len(bins) == 0
            or len(kws) == 0
            or (np.isnan(table).all() if table.size else False)
        )
        title_label = (
            f"Retrieval keyword residuals — {title}" if diverging else f"Retrieval keyword mass — {title}"
        )
        if no_data:
            ax.text(
                0.5,
                0.5,
                f"No data for {title}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title_label)
            ax.set_axis_off()
        elif diverging:
            cmap = plt.get_cmap("RdBu_r").copy()
            # NaN cells (empty bins) render as light grey rather than crashing.
            cmap.set_bad(color="lightgrey", alpha=0.6)
            finite = table[np.isfinite(table)]
            vmax = float(np.abs(finite).max()) if finite.size else 1.0
            if vmax == 0.0:
                vmax = 1.0
            im = ax.imshow(table, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(kws)))
            ax.set_xticklabels(kws, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(bins)))
            ax.set_yticklabels(bins, fontsize=9)
            ax.set_title(title_label)
            fig.colorbar(
                im,
                ax=ax,
                fraction=0.04,
                pad=0.02,
                label="Pearson residual (z-score)",
            )
            # Mark cells with |z| > 2 (the conventional ~95% significance line) with
            # a thin black border so significant deviations are easy to find.
            for i in range(table.shape[0]):
                for j in range(table.shape[1]):
                    val = table[i, j]
                    if np.isfinite(val) and abs(val) > 2.0:
                        ax.add_patch(
                            plt.Rectangle(
                                (j - 0.5, i - 0.5),
                                1,
                                1,
                                fill=False,
                                edgecolor="black",
                                linewidth=1.2,
                            )
                        )
        else:
            im = ax.imshow(table, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(kws)))
            ax.set_xticklabels(kws, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(bins)))
            ax.set_yticklabels(bins, fontsize=9)
            ax.set_title(title_label)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="avg softmax weight")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Heatmap saved to {out_path}")

    def _render_pair(
        bin_to_kw_mass: dict[str, dict[str, float]],
        bin_counts: dict[str, int],
        pop_kw_mass: dict[str, float],
        population_n: int,
        bin_order_hint: Sequence[str] | None,
        title: str,
        axis_key: str,
        *,
        height: float | None = None,
    ) -> dict[str, object]:
        """Render both raw and residual PDFs for one axis; return the result entry.

        Figure width is computed from the actual keyword count after
        ``_pick_bins_and_keywords`` runs (so ``top_n_keywords=None`` —
        "show all" — sizes the figure correctly). ``height`` overrides the
        default 5 in for axes with many rows (e.g. Charlson 17 rows).
        """
        bin_labels, top_keywords = _pick_bins_and_keywords(bin_to_kw_mass, bin_counts, bin_order_hint)
        raw = _normalize_table(bin_to_kw_mass, bin_counts, bin_labels, top_keywords)
        residual = build_pearson_residual_table(
            bin_to_kw_mass,
            bin_counts,
            pop_kw_mass,
            population_n,
            bin_order=bin_labels,
            keyword_order=top_keywords,
        )
        n_kws = max(1, len(top_keywords))
        fig_width = max(10.0, 0.5 * n_kws + 4.0)
        figsize = (fig_width, height if height is not None else 5.0)
        _render_one(
            raw,
            bin_labels,
            top_keywords,
            title,
            output_dir / f"keyword_demographic_{axis_key}.pdf",
            figsize=figsize,
            diverging=False,
        )
        _render_one(
            residual,
            bin_labels,
            top_keywords,
            title,
            output_dir / f"keyword_demographic_{axis_key}_residual.pdf",
            figsize=figsize,
            diverging=True,
        )
        return {
            "table": raw,
            "bins": bin_labels,
            "keywords": top_keywords,
            "residual": residual,
        }

    result: dict[str, dict] = {}

    age_acc = _accumulate_demographic_bin_mass(doc_ids, diff_scores, age_labels, provider)
    result["age"] = _render_pair(*age_acc, AGE_BIN_ORDER, "Age (years)", "age")

    race_acc = _accumulate_demographic_bin_mass(doc_ids, diff_scores, race_labels, provider)
    result["race"] = _render_pair(*race_acc, RACE_BIN_ORDER, "Race/Ethnicity", "race")

    gender_acc = _accumulate_demographic_bin_mass(doc_ids, diff_scores, gender_labels, provider)
    result["gender"] = _render_pair(*gender_acc, sorted(set(gender_labels)), "Gender", "gender")

    # Optional 4th panel: chronic comorbidities (multi-membership rows).
    if comorbidity_frame is not None and comorbidity_categories:
        if comorbidity_frame.height != patient_frame.height:
            raise RuntimeError(
                f"comorbidity_frame rows ({comorbidity_frame.height}) != patient_frame "
                f"rows ({patient_frame.height}); cannot align to the val schema."
            )
        missing_cols = [c for c in comorbidity_categories if c not in comorbidity_frame.columns]
        if missing_cols:
            raise ValueError(f"comorbidity_frame is missing columns for categories: {missing_cols}")
        comorbidity_mask = np.column_stack(
            [comorbidity_frame[cat].to_numpy().astype(bool) for cat in comorbidity_categories]
        )
        chronic_acc = _accumulate_comorbidity_bin_mass(
            doc_ids,
            diff_scores,
            comorbidity_mask,
            list(comorbidity_categories),
            provider,
            include_none=True,
        )
        # Bin order: requested categories first, then the "None of the tracked"
        # bucket; the helper filters out empty bins.
        chronic_bin_order = [*comorbidity_categories, "None of the tracked"]
        chronic_height = max(8.0, 0.3 * max(1, len(chronic_bin_order)) + 2.0)
        result["chronic"] = _render_pair(
            *chronic_acc,
            chronic_bin_order,
            "Charlson Comorbidity Index",
            "chronic",
            height=chronic_height,
        )

    # Long-format CSV dump of every (axis, bin, keyword) → (raw mass, z-score)
    # cell. Lets the user inspect / sort / cite residuals as numbers without
    # eyeballing PDF cells.
    write_residual_csv(result, output_dir / "keyword_demographic_residuals.csv")

    return result
