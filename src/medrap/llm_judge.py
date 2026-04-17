"""LLM-as-a-judge patient-level retrieval-relevance evaluation.

This module consumes extraction artifacts from a trained MedRAP run
(see :mod:`medrap.extraction`) and produces paper-grade evidence about
*whether* retrieved documents are relevant to the patient they were
retrieved for — complementing the existing diagnostics that only show
*what* is retrieved.

Four comparison families are defined. For each sampled anchor patient we
construct one or more pairs of (target_doc, other_doc) and ask an LLM to
pick which document is more relevant for predicting a specified clinical
outcome for this specific patient:

==  =======================================================  ==================
ID  Description                                              Other doc source
==  =======================================================  ==================
F1  retrieved-vs-random                                      random doc
F2  high-rank-vs-low-rank (same patient)                     patient top-`j`
F3  retrieved-vs-same-label-other-patient                    other patient top-1
F4  retrieved-vs-opposite-label-other-patient                other patient top-1
==  =======================================================  ==================

The headline metric is the ``target_preferred_rate`` per family, with
standard error and 95% confidence intervals produced by a
**patient-cluster bootstrap** that resamples patients (not pairs).

The ``openai`` and ``xlsxwriter`` packages are imported lazily inside
the classes/functions that need them so this module (and its doctests)
can be collected without the optional ``llm_judge`` extra installed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import datetime


SYSTEM_PROMPT = (
    "You are a clinical research assistant judging which of two reference "
    "documents is more relevant for predicting a specific clinical outcome "
    "for a specific patient. You will receive (1) a one-sentence task "
    "description, (2) a compact event-by-event timeline for the patient up "
    "to the prediction time, and (3) two candidate documents labeled "
    "DOCUMENT A and DOCUMENT B. Decide which document would be more useful "
    "to a clinician reasoning about the outcome for THIS patient. Base your "
    "decision only on the patient's clinical presentation and the documents' "
    "content — not on writing style or length. Respond with a single JSON "
    'object matching the schema: {"winner": "A"|"B"|"tie", "confidence": '
    '0..1, "rationale": "<=1 sentence"}.'
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JudgePair:
    """One (target_doc, other_doc) pair for LLM evaluation."""

    pair_id: str
    family: str
    anchor_row_idx: int
    anchor_subject_id: int
    anchor_label: int
    target_doc_id: int
    other_doc_id: int
    target_position: Literal["A", "B"]
    other_source_row_idx: int | None = None
    other_source_subject_id: int | None = None
    other_rank: int | None = None
    rng_seed: int = 0


@dataclass(frozen=True, slots=True)
class Verdict:
    """One LLM verdict for a :class:`JudgePair`."""

    pair_id: str
    winner_position: Literal["A", "B", "tie", "invalid"]
    target_won: bool | None
    confidence: float
    rationale: str
    raw_response: str
    model: str
    prompt_tokens: int
    completion_tokens: int


# ---------------------------------------------------------------------------
# Judge protocol + implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class Judge(Protocol):
    """Anything that can answer a pairwise relevance question."""

    def judge(self, system_prompt: str, user_prompt: str, *, seed: int) -> Verdict: ...


class OpenAIJudge:
    """OpenAI-backed :class:`Judge` using Structured Outputs.

    Never raises on API errors — returns a :class:`Verdict` with
    ``winner_position="invalid"`` so one flaky call doesn't crash a
    400-call run.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client: Any | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = client

    def judge(self, system_prompt: str, user_prompt: str, *, seed: int) -> Verdict:
        import json

        client = self._client
        if client is None:
            try:
                from openai import OpenAI

                client = OpenAI()
                self._client = client
            except Exception as e:
                return Verdict(
                    pair_id="",
                    winner_position="invalid",
                    target_won=None,
                    confidence=0.0,
                    rationale=f"openai client init failed: {e}",
                    raw_response="",
                    model=self.model,
                    prompt_tokens=0,
                    completion_tokens=0,
                )

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "judge_verdict",
                "schema": {
                    "type": "object",
                    "properties": {
                        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"},
                    },
                    "required": ["winner", "confidence", "rationale"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        try:
            resp = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                seed=seed,
                response_format=response_format,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        except Exception as e:
            return Verdict(
                pair_id="",
                winner_position="invalid",
                target_won=None,
                confidence=0.0,
                rationale=f"api error: {e}",
                raw_response="",
                model=self.model,
                prompt_tokens=0,
                completion_tokens=0,
            )

        try:
            parsed = json.loads(content)
            winner = parsed.get("winner", "invalid")
            if winner not in ("A", "B", "tie"):
                winner = "invalid"
            confidence = float(parsed.get("confidence", 0.0))
            rationale = str(parsed.get("rationale", ""))
        except Exception as e:
            return Verdict(
                pair_id="",
                winner_position="invalid",
                target_won=None,
                confidence=0.0,
                rationale=f"parse error: {e}",
                raw_response=content,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        return Verdict(
            pair_id="",
            winner_position=winner,  # type: ignore[arg-type]
            target_won=None,
            confidence=confidence,
            rationale=rationale,
            raw_response=content,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class FakeJudge:
    """Test-only :class:`Judge` that returns canned verdicts.

    Constructors:

    - :meth:`FakeJudge.always_A` — always picks slot A.
    - :meth:`FakeJudge.always_target` — always picks the target slot for a
      pair (requires a ``pair_lookup`` keyed by pair rng_seed).
    - :meth:`FakeJudge.flaky` — alternates winners deterministically.
    """

    def __init__(self, rule: Callable[[str, str, int], Verdict]) -> None:
        self._rule = rule

    def judge(self, system_prompt: str, user_prompt: str, *, seed: int) -> Verdict:
        return self._rule(system_prompt, user_prompt, seed)

    @classmethod
    def always_A(cls) -> FakeJudge:  # noqa: N802 - mirrors slot name
        def rule(_sys: str, _user: str, _seed: int) -> Verdict:
            return Verdict(
                pair_id="",
                winner_position="A",
                target_won=None,
                confidence=1.0,
                rationale="always_A",
                raw_response='{"winner":"A","confidence":1.0,"rationale":"always_A"}',
                model="fake",
                prompt_tokens=0,
                completion_tokens=0,
            )

        return cls(rule)

    @classmethod
    def always_target(cls, pair_lookup: dict[int, JudgePair]) -> FakeJudge:
        def rule(_sys: str, _user: str, seed: int) -> Verdict:
            pair = pair_lookup[seed]
            winner = pair.target_position
            return Verdict(
                pair_id=pair.pair_id,
                winner_position=winner,
                target_won=None,
                confidence=1.0,
                rationale="always_target",
                raw_response=f'{{"winner":"{winner}","confidence":1.0,"rationale":"always_target"}}',
                model="fake",
                prompt_tokens=0,
                completion_tokens=0,
            )

        return cls(rule)

    @classmethod
    def flaky(cls, seed: int) -> FakeJudge:
        rng = np.random.default_rng(seed)

        def rule(_sys: str, _user: str, _seed: int) -> Verdict:
            pick = "A" if rng.random() < 0.5 else "B"
            return Verdict(
                pair_id="",
                winner_position=pick,  # type: ignore[arg-type]
                target_won=None,
                confidence=0.5,
                rationale="flaky",
                raw_response=f'{{"winner":"{pick}"}}',
                model="fake",
                prompt_tokens=0,
                completion_tokens=0,
            )

        return cls(rule)


# ---------------------------------------------------------------------------
# Patient timeline rendering + prompt construction
# ---------------------------------------------------------------------------


class PatientTimelineRenderer:
    """Render a patient's MEDS code sequence as human-readable text.

    Loads ``codes.parquet`` once into a ``code → description`` dict and
    uses it to annotate each event. The codes file must be a
    **1-to-1 dictionary** (unique ``code`` values) — this is enforced at
    construction time because an event-level file would silently leak
    label-dependent information into the judge prompt.
    """

    def __init__(
        self,
        *,
        codes_parquet: Path,
        max_events: int = 150,
        include_description: bool = True,
    ) -> None:
        self.codes_parquet = Path(codes_parquet)
        self.max_events = max_events
        self.include_description = include_description

        df = pl.read_parquet(self.codes_parquet, columns=["code", "description"])

        counts = df.group_by("code").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
        if counts.height > 0:
            dups = counts["code"].to_list()
            raise ValueError(
                f"codes.parquet must have unique 'code' values (1-to-1 code→description "
                f"dictionary). Found {counts.height} duplicated codes, first: {dups[0]}. "
                f"Refusing to load an event-level or ambiguous metadata file."
            )

        self._code_to_description: dict[str, str | None] = {
            c: d for c, d in zip(df["code"].to_list(), df["description"].to_list(), strict=True)
        }

    def render(
        self,
        subject_id: int,
        prediction_time: datetime,
        meds_cohort_dir: Path,
    ) -> str:
        parquet_glob = str(Path(meds_cohort_dir) / "data" / "*" / "*.parquet")
        events = (
            pl.scan_parquet(parquet_glob)
            .filter((pl.col("subject_id") == int(subject_id)) & (pl.col("time") <= prediction_time))
            .select(["time", "code"])
            .sort("time")
            .collect()
        )
        events = events.tail(self.max_events)
        lines: list[str] = []
        for row in events.iter_rows(named=True):
            code = row["code"]
            if self.include_description:
                desc = self._code_to_description.get(code)
                if desc:
                    lines.append(f"{code} — {desc}")
                else:
                    lines.append(code)
            else:
                lines.append(code)
        return "\n".join(lines)


class JudgePromptBuilder:
    """Build the (system_prompt, user_prompt) pair for one :class:`JudgePair`."""

    def __init__(
        self,
        *,
        task_description: str,
        timeline_renderer: PatientTimelineRenderer,
        retrieval_ds: Any,
        doc_text_column: str = "doc_text",
        doc_id_to_row: dict[int, int] | None = None,
        max_doc_chars: int = 4000,
    ) -> None:
        self.task_description = task_description
        self.timeline_renderer = timeline_renderer
        self.retrieval_ds = retrieval_ds
        self.doc_text_column = doc_text_column
        self.doc_id_to_row = dict(doc_id_to_row) if doc_id_to_row is not None else {}
        self.max_doc_chars = max_doc_chars

    def _doc_text(self, doc_id: int) -> str:
        row = self.doc_id_to_row.get(int(doc_id))
        if row is None:
            return f"[document id={doc_id} not available]"
        text = self.retrieval_ds[int(row)][self.doc_text_column]
        if text is None:
            return ""
        return str(text)[: self.max_doc_chars]

    def build(self, pair: JudgePair, patient_timeline: str) -> tuple[str, str]:
        target_text = self._doc_text(pair.target_doc_id)
        other_text = self._doc_text(pair.other_doc_id)
        if pair.target_position == "A":
            doc_a, doc_b = target_text, other_text
        else:
            doc_a, doc_b = other_text, target_text
        user_prompt = (
            f"TASK: {self.task_description}\n\n"
            f"PATIENT (sequence of MEDS codes up to prediction time):\n{patient_timeline}\n\n"
            f"DOCUMENT A:\n{doc_a}\n\n"
            f"DOCUMENT B:\n{doc_b}\n\n"
            "Which document (A or B) is more relevant? Respond with "
            '{"winner": "A"|"B"|"tie", "confidence": 0..1, "rationale": "<=1 sentence"}.'
        )
        return SYSTEM_PROMPT, user_prompt


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------


def _stratified_anchor_sample(
    labels: np.ndarray, n_patients: int, rng: np.random.Generator
) -> np.ndarray:
    """50/50-by-label anchor selection, clamped to what's available."""
    pos_pool = np.where(labels == 1)[0]
    neg_pool = np.where(labels == 0)[0]
    n_per_side = n_patients // 2
    n_pos = int(min(n_per_side, len(pos_pool)))
    n_neg = int(min(n_patients - n_pos, len(neg_pool)))
    selected_pos = rng.choice(pos_pool, size=n_pos, replace=False) if n_pos else np.array([], dtype=int)
    selected_neg = rng.choice(neg_pool, size=n_neg, replace=False) if n_neg else np.array([], dtype=int)
    anchors = np.concatenate([selected_pos, selected_neg])
    rng.shuffle(anchors)
    return anchors


def build_pairs(
    *,
    artifacts: dict[str, Any],
    val_schema: pl.DataFrame,
    labels: np.ndarray,
    families: Sequence[str] = ("F1", "F2", "F3", "F4"),
    n_patients: int = 100,
    pairs_per_patient_per_family: int = 1,
    corpus_size: int,
    k: int,
    seed: int = 42,
    dedupe_identical_docs: bool = True,
    skip_missing_families: bool = True,
) -> list[JudgePair]:
    """Construct the frozen list of pairs for this evaluation run.

    See ``D3_plan.md`` for family semantics.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    doc_ids = np.asarray(artifacts["doc_ids"])
    if doc_ids.ndim != 3:
        raise ValueError(f"doc_ids must be (N, R, K); got {doc_ids.shape}")

    subject_ids = val_schema["subject_id"].to_numpy()
    anchors = _stratified_anchor_sample(labels, n_patients, rng)

    def _sample_other_row(pool: np.ndarray, target_doc: int) -> tuple[int, int] | None:
        """Pick a row from pool whose top-1 != target_doc. None if dedupe fails."""
        if len(pool) == 0:
            return None
        for _ in range(10):
            row = int(rng.choice(pool))
            candidate = int(doc_ids[row, 0, 0])
            if not dedupe_identical_docs or candidate != target_doc:
                return row, candidate
        return None

    pairs: list[JudgePair] = []
    counter = 0

    for anchor_idx in anchors:
        anchor_idx = int(anchor_idx)
        anchor_label = int(labels[anchor_idx])
        anchor_sid = int(subject_ids[anchor_idx])
        target_doc = int(doc_ids[anchor_idx, 0, 0])

        for family in families:
            if family == "F2" and k < 2:
                if skip_missing_families:
                    continue
                raise ValueError("F2 requires k >= 2")

            for _ in range(pairs_per_patient_per_family):
                other_doc: int | None = None
                other_source_row: int | None = None
                other_source_sid: int | None = None
                other_rank: int | None = None

                if family == "F1":
                    for _ in range(10):
                        candidate = int(rng.integers(corpus_size))
                        if not dedupe_identical_docs or candidate != target_doc:
                            other_doc = candidate
                            break
                elif family == "F2":
                    j = int(rng.integers(1, k))
                    candidate = int(doc_ids[anchor_idx, 0, j])
                    if dedupe_identical_docs and candidate == target_doc:
                        continue  # unusual: top-1 == top-j; skip this pair
                    other_doc = candidate
                    other_rank = j + 1
                elif family == "F3":
                    pool = np.where(labels == anchor_label)[0]
                    pool = pool[pool != anchor_idx]
                    result = _sample_other_row(pool, target_doc)
                    if result is None:
                        continue
                    other_source_row, other_doc = result
                    other_source_sid = int(subject_ids[other_source_row])
                elif family == "F4":
                    pool = np.where(labels == (1 - anchor_label))[0]
                    result = _sample_other_row(pool, target_doc)
                    if result is None:
                        continue
                    other_source_row, other_doc = result
                    other_source_sid = int(subject_ids[other_source_row])
                else:
                    raise ValueError(f"Unknown family: {family!r}")

                if other_doc is None:
                    continue

                target_position: Literal["A", "B"] = "A" if rng.random() < 0.5 else "B"
                rng_seed = int(rng.integers(1 << 30))
                counter += 1
                pairs.append(
                    JudgePair(
                        pair_id=f"p{counter:06d}",
                        family=family,
                        anchor_row_idx=anchor_idx,
                        anchor_subject_id=anchor_sid,
                        anchor_label=anchor_label,
                        target_doc_id=target_doc,
                        other_doc_id=other_doc,
                        target_position=target_position,
                        other_source_row_idx=other_source_row,
                        other_source_subject_id=other_source_sid,
                        other_rank=other_rank,
                        rng_seed=rng_seed,
                    )
                )
    return pairs


# ---------------------------------------------------------------------------
# Runner + aggregation
# ---------------------------------------------------------------------------


def _compute_target_won(winner_position: str, target_position: str) -> bool | None:
    if winner_position == "A" or winner_position == "B":
        return winner_position == target_position
    return None


def run_judge(
    pairs: Sequence[JudgePair],
    *,
    judge: Judge,
    prompt_builder: JudgePromptBuilder,
    timelines_by_subject_id: dict[int, str] | None = None,
    max_workers: int = 8,
    progress: bool = True,
) -> pl.DataFrame:
    """Call the judge for every pair and return a long-form DataFrame.

    Timelines are optionally pre-rendered and passed in keyed by
    ``anchor_subject_id``. If missing for a subject, an empty string is
    used (unit tests rely on this).
    """
    timelines = timelines_by_subject_id or {}

    def _run_one(pair: JudgePair) -> tuple[JudgePair, Verdict]:
        timeline = timelines.get(pair.anchor_subject_id, "")
        sys_prompt, user_prompt = prompt_builder.build(pair, patient_timeline=timeline)
        verdict = judge.judge(sys_prompt, user_prompt, seed=pair.rng_seed)
        return pair, verdict

    if max_workers > 1 and len(pairs) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_run_one, pairs))
    else:
        results = [_run_one(p) for p in pairs]

    rows = []
    for pair, verdict in results:
        rows.append(
            {
                "pair_id": pair.pair_id,
                "family": pair.family,
                "anchor_subject_id": pair.anchor_subject_id,
                "anchor_row_idx": pair.anchor_row_idx,
                "anchor_label": pair.anchor_label,
                "target_doc_id": pair.target_doc_id,
                "other_doc_id": pair.other_doc_id,
                "target_position": pair.target_position,
                "other_rank": pair.other_rank,
                "other_source_subject_id": pair.other_source_subject_id,
                "winner_position": verdict.winner_position,
                "target_won": _compute_target_won(verdict.winner_position, pair.target_position),
                "confidence": verdict.confidence,
                "rationale": verdict.rationale,
                "model": verdict.model,
                "prompt_tokens": verdict.prompt_tokens,
                "completion_tokens": verdict.completion_tokens,
                "raw_response": verdict.raw_response,
            }
        )
    return pl.DataFrame(rows)


def summarize_winrates(
    df: pl.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
    ci_level: float = 0.95,
    invalid_policy: Literal["drop", "count_as_loss"] = "drop",
) -> pl.DataFrame:
    """Compute per-family ``target_preferred_rate`` with patient-cluster bootstrap CI.

    Algorithm:

    1. Count invalid verdicts (``target_won is None``) — surfaced as ``n_invalid``.
    2. Apply ``invalid_policy``: ``"drop"`` removes invalid rows;
       ``"count_as_loss"`` converts their ``target_won`` to ``False``.
    3. Within-patient averaging first — ``per_patient = mean(target_won)`` per
       ``(family, anchor_subject_id)`` — so a patient's pairs count once.
    4. Point estimate = ``mean(per_patient)`` across patients in the family.
    5. Patient-cluster bootstrap: resample ``N`` patients with replacement from
       the per-patient means, ``n_bootstrap`` times; SE = ``np.std(replicates, ddof=1)``;
       CI via percentile method at ``ci_level``.
    """
    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci_level) / 2.0

    results: list[dict[str, Any]] = []
    for family in sorted(df["family"].unique().to_list()):
        fam_df = df.filter(pl.col("family") == family)
        n_pairs = fam_df.height
        n_invalid = fam_df.filter(pl.col("target_won").is_null()).height

        if invalid_policy == "drop":
            working = fam_df.filter(pl.col("target_won").is_not_null())
        else:  # count_as_loss
            working = fam_df.with_columns(pl.col("target_won").fill_null(False))

        if working.height == 0:
            results.append(
                {
                    "family": family,
                    "n_patients": 0,
                    "n_pairs": n_pairs,
                    "n_invalid": n_invalid,
                    "target_preferred_rate": float("nan"),
                    "standard_error": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "bootstrap_mean": float("nan"),
                }
            )
            continue

        per_patient = (
            working.with_columns(pl.col("target_won").cast(pl.Float64))
            .group_by("anchor_subject_id")
            .agg(pl.col("target_won").mean().alias("mean_target_won"))
        )
        values = np.asarray(per_patient["mean_target_won"].to_list(), dtype=float)
        n_patients = int(values.size)

        point = float(values.mean())

        if n_bootstrap > 0:
            idx = rng.integers(0, n_patients, size=(n_bootstrap, n_patients))
            replicates = values[idx].mean(axis=1)
            se = float(np.std(replicates, ddof=1)) if n_bootstrap > 1 else 0.0
            ci_low = float(np.quantile(replicates, alpha))
            ci_high = float(np.quantile(replicates, 1.0 - alpha))
            boot_mean = float(replicates.mean())
        else:
            se = 0.0
            ci_low = point
            ci_high = point
            boot_mean = point

        results.append(
            {
                "family": family,
                "n_patients": n_patients,
                "n_pairs": n_pairs,
                "n_invalid": n_invalid,
                "target_preferred_rate": point,
                "standard_error": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bootstrap_mean": boot_mean,
            }
        )

    return pl.DataFrame(results)


# ---------------------------------------------------------------------------
# Per-patient rollup and human-validation subset
# ---------------------------------------------------------------------------


def _softmax_positive_prob_and_pred(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (positive-class prob, argmax label) for 2-class logits."""
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 2 and logits.shape[1] >= 2:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs[:, 1], probs.argmax(axis=1).astype(int)
    sig = 1.0 / (1.0 + np.exp(-logits.squeeze()))
    pred = (sig >= 0.5).astype(int)
    return sig, pred


def _age_bin(age: float | None) -> str | None:
    if age is None:
        return None
    if age < 30:
        return "<30"
    if age < 50:
        return "30-49"
    if age < 70:
        return "50-69"
    if age < 90:
        return "70-89"
    return "90+"


def build_per_patient_rollup(
    pairs: Sequence[JudgePair],
    verdicts: pl.DataFrame,
    *,
    logits: np.ndarray,
    targets: np.ndarray,
    artifacts: dict[str, Any],
    timeline_renderer: PatientTimelineRenderer,
    val_schema: pl.DataFrame,
    demographics: pl.DataFrame,
    retrieval_ds: Any,
    doc_id_to_row: dict[int, int],
    doc_text_column: str = "doc_text",
    doc_metadata_columns: Sequence[str] = ("title",),
    doc_text_preview_chars: int = 300,
    timelines_by_subject_id: dict[int, str] | None = None,
    families: Sequence[str] = ("F1", "F2", "F3", "F4"),
) -> pl.DataFrame:
    """One row per sampled patient with per-family outcomes and rich metadata.

    The ``timeline_renderer`` argument is retained for API compatibility; the
    caller is expected to pre-render timelines and pass them via
    ``timelines_by_subject_id`` when it has access to the MEDS cohort dir.
    """
    del timeline_renderer  # kept in the signature per plan; rendering is caller's job
    timelines = timelines_by_subject_id or {}

    ds_columns: set[str] = set(getattr(retrieval_ds, "column_names", []) or [])
    available_meta = [c for c in doc_metadata_columns if c in ds_columns]

    def _doc_fields(doc_id: int | None) -> dict[str, Any]:
        out: dict[str, Any] = {"text_preview": None, "meta": dict.fromkeys(available_meta)}
        if doc_id is None:
            return out
        row = doc_id_to_row.get(int(doc_id))
        if row is None:
            return out
        try:
            rec = retrieval_ds[int(row)]
        except Exception:
            return out
        text = rec.get(doc_text_column, "")
        if isinstance(text, str):
            out["text_preview"] = text[:doc_text_preview_chars]
        for c in available_meta:
            out["meta"][c] = rec.get(c)
        return out

    # Unique anchors in order of first appearance in ``pairs``.
    seen: set[int] = set()
    ordered_anchors: list[tuple[int, int]] = []
    for p in pairs:
        if p.anchor_subject_id not in seen:
            seen.add(p.anchor_subject_id)
            ordered_anchors.append((p.anchor_row_idx, p.anchor_subject_id))

    demo_cols = set(demographics.columns) if demographics is not None else set()
    demo_dict: dict[int, dict[str, Any]] = {}
    if demographics is not None and demographics.height > 0:
        for rec in demographics.iter_rows(named=True):
            demo_dict[int(rec["subject_id"])] = rec

    schema_dict: dict[int, dict[str, Any]] = {}
    for rec in val_schema.iter_rows(named=True):
        schema_dict[int(rec["subject_id"])] = rec

    pos_prob, pred_label = _softmax_positive_prob_and_pred(logits)
    targets_np = np.asarray(targets).astype(int)
    doc_ids_array = np.asarray(artifacts["doc_ids"])
    doc_scores_array = np.asarray(artifacts["doc_scores"])

    rows: list[dict[str, Any]] = []
    for anchor_row_idx, anchor_sid in ordered_anchors:
        schema_rec = schema_dict.get(anchor_sid, {})
        demo = demo_dict.get(anchor_sid, {})
        prediction_time = schema_rec.get("prediction_time")
        birth_time = demo.get("birth_time") if "birth_time" in demo_cols else None
        age_years: float | None = None
        if birth_time is not None and prediction_time is not None:
            age_years = (prediction_time - birth_time).days / 365.25

        row: dict[str, Any] = {
            "anchor_subject_id": anchor_sid,
            "prediction_time": prediction_time,
            "anchor_label": int(targets_np[anchor_row_idx]),
            "predicted_label": int(pred_label[anchor_row_idx]),
            "predicted_prob": float(pos_prob[anchor_row_idx]),
        }
        row["prediction_correct"] = row["predicted_label"] == row["anchor_label"]
        if "gender" in demo_cols:
            row["gender"] = demo.get("gender")
        if "race" in demo_cols:
            row["race"] = demo.get("race")
        row["age_years_at_prediction"] = age_years
        row["age_bin"] = _age_bin(age_years)
        row["patient_timeline"] = timelines.get(anchor_sid, "")

        target_doc_id = int(doc_ids_array[anchor_row_idx, 0, 0])
        row["target_doc_id"] = target_doc_id
        row["target_doc_score"] = float(doc_scores_array[anchor_row_idx, 0, 0])
        target_fields = _doc_fields(target_doc_id)
        row["target_doc_text_preview"] = target_fields["text_preview"]
        for c in available_meta:
            row[f"target_doc_{c}"] = target_fields["meta"][c]

        for fam in families:
            group = verdicts.filter(
                (pl.col("anchor_subject_id") == anchor_sid) & (pl.col("family") == fam)
            )
            if group.height == 0:
                row[f"{fam}_target_won"] = None
                row[f"{fam}_winner_position"] = None
                row[f"{fam}_confidence"] = None
                row[f"{fam}_rationale"] = None
                row[f"{fam}_other_doc_id"] = None
                row[f"{fam}_other_doc_text_preview"] = None
                for c in available_meta:
                    row[f"{fam}_other_doc_{c}"] = None
                if fam == "F2":
                    row["F2_other_rank"] = None
                if fam in ("F3", "F4"):
                    row[f"{fam}_other_source_subject_id"] = None
                continue

            first = group.row(0, named=True)
            valid_tw = group["target_won"].drop_nulls()
            row[f"{fam}_target_won"] = (
                float(valid_tw.cast(pl.Float64).mean()) if valid_tw.len() > 0 else None
            )
            row[f"{fam}_winner_position"] = first.get("winner_position")
            conf = group["confidence"].drop_nulls()
            row[f"{fam}_confidence"] = float(conf.mean()) if conf.len() > 0 else None
            row[f"{fam}_rationale"] = first.get("rationale")

            other_doc_id = first.get("other_doc_id")
            row[f"{fam}_other_doc_id"] = other_doc_id
            of = _doc_fields(int(other_doc_id) if other_doc_id is not None else None)
            row[f"{fam}_other_doc_text_preview"] = of["text_preview"]
            for c in available_meta:
                row[f"{fam}_other_doc_{c}"] = of["meta"][c]
            if fam == "F2":
                row["F2_other_rank"] = first.get("other_rank")
            if fam in ("F3", "F4"):
                row[f"{fam}_other_source_subject_id"] = first.get("other_source_subject_id")

        rows.append(row)

    return pl.DataFrame(rows)


def build_human_validation_subset(
    df: pl.DataFrame,
    *,
    n: int = 50,
    seed: int = 42,
    retrieval_ds: Any,
    doc_id_to_row: dict[int, int],
    doc_metadata_columns: Sequence[str] = ("title",),
) -> pl.DataFrame:
    """Anonymized human-review subset.

    Drops columns that reveal which slot (A or B) held the target document
    (``target_doc_id``, ``target_position``, ``target_won``, ``winner_position``,
    ``other_source_subject_id``, ``other_rank``, ``model``, ``raw_response``,
    ``confidence``, ``rationale``). Re-materializes ``doc_a_text``/``doc_b_text``
    and optional metadata columns (e.g. ``doc_a_title``/``doc_b_title``) so the
    rater can read the documents they're judging. Row order is shuffled with a
    separate seed so the rater can't infer slot-position from sheet order.
    """
    if df.height == 0:
        return df.clone()

    rng = np.random.default_rng(seed)

    families = sorted(df["family"].unique().to_list())
    family_sizes = {f: df.filter(pl.col("family") == f).height for f in families}
    total = sum(family_sizes.values())
    target_n = min(n, total)

    # Proportional allocation with min(5, available) per family.
    floors = {f: min(5, family_sizes[f]) for f in families}
    if sum(floors.values()) > target_n:
        alloc = floors
    else:
        remaining = target_n - sum(floors.values())
        props: dict[str, float] = {
            f: (family_sizes[f] - floors[f]) / max(total - sum(floors.values()), 1)
            for f in families
        }
        extras = {f: int(round(remaining * props[f])) for f in families}
        alloc = {f: min(floors[f] + extras[f], family_sizes[f]) for f in families}

    sampled_frames: list[pl.DataFrame] = []
    for f in families:
        fam_df = df.filter(pl.col("family") == f)
        k = min(alloc[f], fam_df.height)
        if k <= 0:
            continue
        idx = rng.choice(fam_df.height, size=k, replace=False).tolist()
        sampled_frames.append(fam_df[idx])

    if not sampled_frames:
        return df.clone().clear()

    subset = pl.concat(sampled_frames)

    # Re-materialize docs into slot A / slot B according to target_position.
    ds_columns: set[str] = set(getattr(retrieval_ds, "column_names", []) or [])
    available_meta = [c for c in doc_metadata_columns if c in ds_columns]

    def _doc_fields(doc_id: int | None) -> dict[str, Any]:
        out: dict[str, Any] = {"text": "", "meta": {c: None for c in available_meta}}
        if doc_id is None:
            return out
        row = doc_id_to_row.get(int(doc_id))
        if row is None:
            return out
        try:
            rec = retrieval_ds[int(row)]
        except Exception:
            return out
        out["text"] = str(rec.get("doc_text", ""))
        for c in available_meta:
            out["meta"][c] = rec.get(c)
        return out

    target_ids = subset["target_doc_id"].to_list()
    other_ids = subset["other_doc_id"].to_list()
    target_pos = subset["target_position"].to_list()

    doc_a_texts: list[str] = []
    doc_b_texts: list[str] = []
    doc_a_metas: dict[str, list[Any]] = {c: [] for c in available_meta}
    doc_b_metas: dict[str, list[Any]] = {c: [] for c in available_meta}

    for tid, oid, pos in zip(target_ids, other_ids, target_pos, strict=True):
        tgt = _doc_fields(tid)
        oth = _doc_fields(oid)
        if pos == "A":
            a, b = tgt, oth
        else:
            a, b = oth, tgt
        doc_a_texts.append(a["text"])
        doc_b_texts.append(b["text"])
        for c in available_meta:
            doc_a_metas[c].append(a["meta"][c])
            doc_b_metas[c].append(b["meta"][c])

    banned = {
        "target_doc_id",
        "other_doc_id",
        "target_position",
        "target_won",
        "winner_position",
        "other_source_subject_id",
        "other_rank",
        "model",
        "raw_response",
        "confidence",
        "rationale",
        "prompt_tokens",
        "completion_tokens",
        "anchor_row_idx",
    }
    keep_cols = [c for c in subset.columns if c not in banned]
    out = subset.select(keep_cols).with_columns(
        pl.Series("doc_a_text", doc_a_texts),
        pl.Series("doc_b_text", doc_b_texts),
    )
    for c in available_meta:
        out = out.with_columns(
            pl.Series(f"doc_a_{c}", doc_a_metas[c]),
            pl.Series(f"doc_b_{c}", doc_b_metas[c]),
        )

    out = out.with_columns(
        pl.Series("human_winner", [None] * out.height, dtype=pl.Utf8),
        pl.Series("human_confidence", [None] * out.height, dtype=pl.Int64),
        pl.Series("human_notes", [None] * out.height, dtype=pl.Utf8),
    )

    shuffle_rng = np.random.default_rng(seed + 1)
    perm = shuffle_rng.permutation(out.height).tolist()
    return out[perm]


# ---------------------------------------------------------------------------
# Excel writer
# ---------------------------------------------------------------------------


def write_results_workbook(
    path: Path,
    *,
    family_winrates: pl.DataFrame,
    per_patient: pl.DataFrame,
    pairs_verdicts: pl.DataFrame,
    human_validation: pl.DataFrame,
) -> None:
    """Write a 4-sheet ``.xlsx`` workbook via ``xlsxwriter`` (lazy import).

    Sheet names are fixed: ``family_winrates``, ``per_patient_results``,
    ``pairs_verdicts``, ``human_validation``. Requires the ``llm_judge`` extra.
    """
    try:
        import xlsxwriter
    except ImportError as e:
        raise ImportError(
            "xlsxwriter is required to write the results workbook. "
            "Install with `pip install medrap[llm_judge]`."
        ) from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sheets: list[tuple[str, pl.DataFrame]] = [
        ("family_winrates", family_winrates),
        ("per_patient_results", per_patient),
        ("pairs_verdicts", pairs_verdicts),
        ("human_validation", human_validation),
    ]

    with xlsxwriter.Workbook(str(path)) as wb:
        for sheet_name, frame in sheets:
            frame.write_excel(workbook=wb, worksheet=sheet_name)
