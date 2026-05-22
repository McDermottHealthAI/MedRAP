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

import re
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
    "You are a board-certified clinician acting as a judge in a retrieval "
    "evaluation. You will receive, inside XML tags:\n"
    "  <task>    the clinical prediction task\n"
    "  <patient> a compact summary of one patient up to the prediction time\n"
    "  <document_a>, <document_b>  two candidate reference documents\n\n"
    "Your job is to decide which document — A or B — a clinician reasoning "
    "about the task for THIS specific patient would find more useful.\n\n"
    "EVALUATION CRITERIA (in priority order):\n"
    "  1. Patient specificity: does the document address the patient's actual "
    "clinical picture (conditions, meds, labs, procedures), not just the "
    "disease area?\n"
    "  2. Task alignment: does the document's content inform the specific "
    "outcome being predicted?\n"
    "  3. Decision utility: would a clinician pull this document off a shelf "
    "at the bedside for this patient?\n"
    "Prefer concrete patient-relevant content over generic or tangentially "
    "related material. If both documents are equally (ir)relevant, return "
    '"tie".\n\n'
    "BIAS CONTROLS — ignore these when judging:\n"
    "  - Document length, writing style, formality\n"
    "  - Position (A vs B) — the assignment is randomized\n"
    "  - Title or source name alone (judge on content)\n\n"
    "Respond with a SINGLE JSON object and nothing else, matching the schema:\n"
    '  {"winner": "A" | "B" | "tie", '
    '"confidence": number in [0, 1], '
    '"rationale": string (<=1 sentence citing patient detail + doc content)}'
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


_TYPE_LABELS: dict[str, str] = {
    "LAB": "Lab",
    "PROCEDURE": "Procedure",
    "MEDICATION": "Medication",
    "DIAGNOSIS": "Diagnosis",
    "SUBJECT_FLUID_OUTPUT": "Fluid output",
    "HOSPITAL_ADMISSION": "Hospital admission",
    "ICU_ADMISSION": "ICU admission",
    "MEDS_BIRTH": "Birth",
    "GENDER": "Gender",
    "RACE": "Race",
}


def _type_label(code: str) -> str:
    """Map a MEDS code to a human-readable event-type prefix."""
    head = code.split("//", 1)[0]
    return _TYPE_LABELS.get(head, head.replace("_", " ").capitalize())


def _procedure_action(code: str) -> str | None:
    """Return 'start'/'end' for PROCEDURE//START/END codes, else None."""
    parts = code.split("//")
    if len(parts) >= 2 and parts[0] == "PROCEDURE" and parts[1] in ("START", "END"):
        return parts[1].lower()
    return None


_NULL_UNIT_VALUES = frozenset({"UNK", "N/A", "NA", "NONE", "NULL", "-"})


def _unit_from_code(code: str) -> str | None:
    """Extract the unit slot (third ``//`` segment), or ``None`` when it is
    missing / a sentinel for "unknown" / a bare numeric ID.

    The third slot is a true unit only for LAB-style codes (``LAB//item//mg/dL``).
    For codes like ``PROCEDURE//END//225459`` the third slot is an item ID,
    not a unit — we reject purely-numeric slots to avoid treating them as units.
    """
    parts = code.split("//")
    if len(parts) < 3:
        return None
    unit = parts[2].strip()
    if not unit or unit.upper() in _NULL_UNIT_VALUES:
        return None
    # Reject purely-numeric slots (item IDs leaking through as units).
    stripped = unit.replace(".", "").replace("-", "").replace("+", "").replace("/", "")
    if stripped.isdigit():
        return None
    return unit


# LOINC long-names in MIMIC codes.parquet carry measurement-property
# qualifiers in square brackets (``[Moles/volume]``, ``[Mass/volume]``,
# ``[Partial pressure]`` …) plus specimen/method suffixes (``in Serum or
# Plasma``, ``by calculation``, ``by Automated count`` …) that are redundant
# with the unit string and add noise without signal. :func:`_clean_lab_description`
# strips both so the prompt reads like a bedside lab list.
_LAB_DESC_BRACKETS = re.compile(r"\s*\[[^\]]*\]")
_LAB_DESC_SUFFIXES: tuple[str, ...] = (
    " in Serum or Plasma",
    " in Serum",
    " in Plasma",
    " in Blood by calculation",
    " in Blood by Automated count",
    " in Blood by Manual count",
    " in Blood",
    " of Blood",
    " in Urine",
    " in Arterial blood",
    " in Venous blood",
    " by calculation",
    " by Automated count",
    " by Manual count",
)


def _clean_lab_description(desc: str) -> str:
    """Strip LOINC bracketed qualifiers and specimen/method suffixes.

    Examples::

        "Carbon dioxide, total [Moles/volume] in Blood by calculation"
          -> "Carbon dioxide, total"
        "Lactate [Moles/volume] in Blood"  -> "Lactate"
        "pH of Blood"                      -> "pH"
        "Oxygen [Partial pressure] in Blood" -> "Oxygen"
    """
    cleaned = _LAB_DESC_BRACKETS.sub("", desc)
    cleaned = " ".join(cleaned.split())
    changed = True
    while changed:
        changed = False
        for suffix in _LAB_DESC_SUFFIXES:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
                changed = True
    return cleaned.rstrip(",").strip()


def _render_patient_narrative(
    demographics: dict[str, Any] | None,
    prediction_time: datetime | None,
    clinical_summary: dict[str, Any] | None,
) -> str | None:
    """Collapse demographics + clinical summary into a single prose paragraph.

    Used by :meth:`PatientTimelineRenderer.render_categorical`. Matches the
    case-report style GPT-4o/4o-mini see in training: HPI-style opening
    sentence, utilization as a short enumeration, chronic conditions inline.
    """
    sentences: list[str] = []

    if demographics:
        age_str: str | None = None
        birth = demographics.get("birth_time")
        if birth is not None and prediction_time is not None:
            try:
                age_years = (prediction_time - birth).days // 365
                if age_years >= 0:
                    age_str = f"{age_years}-year-old"
            except (TypeError, ValueError):
                age_str = None

        gender_raw = demographics.get("gender")
        gender_str: str | None = None
        if gender_raw:
            g = str(gender_raw).strip().upper()
            gender_str = {"M": "man", "F": "woman"}.get(g, str(gender_raw).lower())

        race_raw = demographics.get("race")
        race_str: str | None = None
        if race_raw:
            race_str = str(race_raw).replace("/", " / ").title()

        head_parts: list[str] = []
        if age_str:
            head_parts.append(age_str)
        if race_str:
            head_parts.append(race_str)
        if gender_str:
            head_parts.append(gender_str)
        if head_parts:
            sentences.append("A " + " ".join(head_parts) + ".")

    if clinical_summary:
        util_bits: list[str] = []
        n_hosp = int(clinical_summary.get("n_hospital_admissions", 0) or 0)
        n_icu = int(clinical_summary.get("n_icu_admissions", 0) or 0)
        n_ed = int(clinical_summary.get("n_ed_visits", 0) or 0)
        if n_hosp:
            util_bits.append(
                f"{n_hosp} hospital admission" + ("s" if n_hosp != 1 else "")
            )
        if n_icu:
            util_bits.append(
                f"{n_icu} ICU stay" + ("s" if n_icu != 1 else "")
            )
        if n_ed:
            util_bits.append(
                f"{n_ed} ED visit" + ("s" if n_ed != 1 else "")
            )
        if util_bits:
            sentences.append("Prior utilization: " + ", ".join(util_bits) + ".")

        conditions = clinical_summary.get("chronic_conditions") or []
        if conditions:
            sentences.append(
                "History notable for " + ", ".join(conditions) + "."
            )
        else:
            sentences.append("No chronic conditions detected on ICD-10 history.")

    if not sentences:
        return None
    return " ".join(sentences)


def _format_numeric(value: float) -> str:
    """Format a numeric lab value concisely (strip trailing zeros)."""
    if value != value:  # NaN
        return ""
    if abs(value) >= 100 or float(value).is_integer():
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _render_demographic_block(
    demographics: dict[str, Any] | None,
    prediction_time: datetime | None,
) -> str | None:
    """Render the leading 'PATIENT: ...' line, or None if nothing to show."""
    if not demographics:
        return None
    parts: list[str] = []

    birth = demographics.get("birth_time")
    age_str: str | None = None
    if birth is not None and prediction_time is not None:
        try:
            age_years = (prediction_time - birth).days // 365
            if age_years >= 0:
                age_str = f"{age_years}-year-old"
        except (TypeError, ValueError):
            age_str = None

    gender_raw = demographics.get("gender")
    gender_str: str | None = None
    if gender_raw:
        g = str(gender_raw).strip().upper()
        gender_str = {"M": "male", "F": "female"}.get(g, str(gender_raw).lower())

    if age_str and gender_str:
        parts.append(f"{age_str} {gender_str}")
    elif age_str:
        parts.append(age_str)
    elif gender_str:
        parts.append(gender_str)

    race = demographics.get("race")
    if race:
        parts.append(f"race {race}")

    if prediction_time is not None:
        parts.append(f"prediction time {prediction_time}")

    if not parts:
        return None
    return "PATIENT: " + ", ".join(parts) + "."


# Simplified Charlson-style chronic-condition flag set. Each key is a
# display name emitted into the judge prompt / rollup; each value is the
# set of ICD-10 code prefixes (without the ``DIAGNOSIS//ICD//10//``
# MEDS prefix) that count as a hit. MIMIC-IV does not ship pre-computed
# comorbidity scores, so we derive flags from the patient's diagnosis
# history (see https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses_icd/).
_CHRONIC_CONDITION_ICD10_PREFIXES: dict[str, tuple[str, ...]] = {
    "Ischemic heart disease / MI": ("I20", "I21", "I22", "I23", "I24", "I25"),
    "Congestive heart failure": ("I50", "I110", "I130", "I132", "I43"),
    "Cerebrovascular disease": (
        "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69",
        "G45", "G46",
    ),
    "Peripheral vascular disease": ("I70", "I71", "I72", "I73", "I74", "I77", "I79"),
    "Chronic pulmonary disease / COPD": (
        "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47",
    ),
    "Diabetes mellitus": ("E10", "E11", "E12", "E13", "E14"),
    "Chronic kidney disease": ("N18", "N19"),
    "Liver disease": ("K70", "K71", "K72", "K73", "K74", "K75", "K76", "K77", "B18"),
    "Malignancy": tuple(f"C{i:02d}" for i in range(0, 98)),
    "AIDS/HIV": ("B20", "B21", "B22", "B24"),
}

_DIAGNOSIS_ICD10_PREFIX = "DIAGNOSIS//ICD//10//"


def compute_patient_clinical_summary(
    subject_id: int,
    prediction_time: datetime,
    meds_cohort_dir: Path,
) -> dict[str, Any]:
    """Derive healthcare-utilization counts + chronic-condition flags from MEDS events.

    Scans events with ``time <= prediction_time`` for the given patient and
    returns a dict with keys:

    - ``n_hospital_admissions``: count of ``HOSPITAL_ADMISSION//...`` codes.
    - ``n_icu_admissions``: count of ``ICU_ADMISSION//...`` codes.
    - ``n_ed_visits``: count of ``TRANSFER_TO//ED...`` codes.
    - ``chronic_conditions``: sorted list of display names from
      :data:`_CHRONIC_CONDITION_ICD10_PREFIXES` that matched at least one
      diagnosis code in the patient's history.
    - ``chronic_condition_count``: ``len(chronic_conditions)``.

    The counts include events from the current admission (the patient's
    ``HOSPITAL_ADMISSION`` event is counted once). MIMIC-IV does not ship
    pre-computed utilization or Charlson/Elixhauser tables — this function
    derives them from the raw ``diagnoses_icd`` + admission/transfer history
    available in the MEDS event stream.
    """
    parquet_glob = str(Path(meds_cohort_dir) / "data" / "*" / "*.parquet")
    lf = pl.scan_parquet(parquet_glob).filter(
        (pl.col("subject_id") == int(subject_id)) & (pl.col("time") <= prediction_time)
    )
    codes = lf.select("code").collect()["code"].to_list()

    n_hosp = 0
    n_icu = 0
    n_ed = 0
    conditions: set[str] = set()
    for code in codes:
        if code is None:
            continue
        if code.startswith("HOSPITAL_ADMISSION//"):
            n_hosp += 1
        elif code.startswith("ICU_ADMISSION//"):
            n_icu += 1
        elif code.startswith("TRANSFER_TO//ED"):
            n_ed += 1
        elif code.startswith(_DIAGNOSIS_ICD10_PREFIX):
            icd = code[len(_DIAGNOSIS_ICD10_PREFIX):]
            for cond_name, prefixes in _CHRONIC_CONDITION_ICD10_PREFIXES.items():
                if cond_name in conditions:
                    continue
                if any(icd.startswith(p) for p in prefixes):
                    conditions.add(cond_name)

    return {
        "n_hospital_admissions": int(n_hosp),
        "n_icu_admissions": int(n_icu),
        "n_ed_visits": int(n_ed),
        "chronic_conditions": sorted(conditions),
        "chronic_condition_count": len(conditions),
    }


def _format_clinical_summary(summary: dict[str, Any]) -> str:
    """Render a :func:`compute_patient_clinical_summary` result for the prompt."""
    util = (
        f"Healthcare utilization (to prediction time): "
        f"{summary['n_hospital_admissions']} hospital admission(s), "
        f"{summary['n_icu_admissions']} ICU stay(s), "
        f"{summary['n_ed_visits']} ED visit(s)"
    )
    conditions = summary.get("chronic_conditions") or []
    chronic = (
        "Chronic conditions (from ICD-10 history): " + ", ".join(conditions)
        if conditions
        else "Chronic conditions (from ICD-10 history): none detected"
    )
    return f"CLINICAL SUMMARY:\n  {util}\n  {chronic}"


class PatientTimelineRenderer:
    """Render a patient's MEDS event sequence as human-readable clinical text.

    The rendered text is built in two parts:

    1. A one-line **demographic header** (age, gender, race, prediction
       time) when a ``demographics`` record is passed to :meth:`render`.
       Always shown even when the static demographic MEDS codes have
       fallen outside the last-N event window.

    2. A **timeline** of the last N *described* events before the
       prediction time. Events whose MEDS ``code`` has no entry in
       ``codes.parquet``'s ``description`` column are **dropped** — they
       are pure noise to the judge (bare MIMIC itemids like
       ``LAB//227944//UNK``). Consecutive duplicates on
       ``(event_type, description)`` are collapsed to a single line (the
       last measurement's numeric value is kept so the LLM sees the most
       recent reading).

    ``codes.parquet`` must be a **1-to-1 dictionary** (unique ``code``
    values) — this is enforced at construction time because an
    event-level metadata file would silently leak label-dependent
    information into the judge prompt.
    """

    def __init__(
        self,
        *,
        codes_parquet: Path,
        max_events: int = 20,
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
        *,
        demographics: dict[str, Any] | None = None,
        clinical_summary: dict[str, Any] | None = None,
    ) -> str:
        parquet_glob = str(Path(meds_cohort_dir) / "data" / "*" / "*.parquet")
        lf = pl.scan_parquet(parquet_glob).filter(
            (pl.col("subject_id") == int(subject_id)) & (pl.col("time") <= prediction_time)
        )
        available = set(lf.collect_schema().names())
        select_cols = ["time", "code"]
        if "numeric_value" in available:
            select_cols.append("numeric_value")
        events = lf.select(select_cols).sort("time").collect()

        # Step 1: transform raw events into rendered rows, dropping those
        # without a description (pure noise to the LLM judge).
        rendered: list[tuple[str, str, str]] = []  # (type_key, description, formatted_line)
        for row in events.iter_rows(named=True):
            code = row["code"]
            desc = self._code_to_description.get(code) if self.include_description else None
            if not desc:
                continue
            # Some MIMIC codes have multi-line descriptions (e.g. a single
            # MEDS itemid maps to several drug preparations, joined by '\n').
            # Collapse to a single line so later lines don't orphan without a
            # type prefix.
            desc = " / ".join(p.strip() for p in str(desc).splitlines() if p.strip())

            type_key = code.split("//", 1)[0]
            label = _type_label(code)
            action = _procedure_action(code)
            if action is not None:
                label = f"{label} ({action})"
            unit = _unit_from_code(code)

            value_part = ""
            value = row.get("numeric_value") if "numeric_value" in row else None
            if value is not None and isinstance(value, (int, float)) and not (
                isinstance(value, float) and value != value  # NaN
            ):
                formatted = _format_numeric(float(value))
                if formatted:
                    value_part = f" = {formatted}"

            unit_part = f" ({unit})" if unit else ""
            line = f"{label}: {desc}{value_part}{unit_part}"
            rendered.append((type_key, desc, line))

        # Step 2: collapse consecutive duplicates on (type, description),
        # keeping the *latest* rendered line (so the most recent numeric
        # value wins).
        collapsed: list[str] = []
        prev_key: tuple[str, str] | None = None
        for type_key, desc, line in rendered:
            key = (type_key, desc)
            if key == prev_key:
                collapsed[-1] = line
            else:
                collapsed.append(line)
                prev_key = key

        # Step 3: keep the last N rows after filter + dedupe.
        tail = collapsed[-self.max_events :] if self.max_events > 0 else collapsed

        # Step 4: assemble demographic header + clinical summary + timeline.
        blocks: list[str] = []
        header = _render_demographic_block(demographics, prediction_time)
        if header is not None:
            blocks.append(header)
        if clinical_summary is not None:
            blocks.append(_format_clinical_summary(clinical_summary))
        if tail:
            blocks.append(
                f"TIMELINE (most recent {len(tail)} described events before prediction):"
            )
            blocks.extend(tail)
        return "\n".join(blocks)

    def render_categorical(
        self,
        subject_id: int,
        prediction_time: datetime,
        meds_cohort_dir: Path,
        *,
        demographics: dict[str, Any] | None = None,
        clinical_summary: dict[str, Any] | None = None,
        max_diagnoses: int = 30,
        max_medications: int = 15,
        max_procedures: int = 15,
        max_labs: int = 15,
    ) -> str:
        """Render the patient as a compact, deduped, category-grouped summary.

        Unlike :meth:`render` (chronological event-by-event), this emits four
        sections — ``DIAGNOSES``, ``ACTIVE MEDICATIONS``, ``RECENT PROCEDURES``,
        ``RECENT LABS`` — each deduped by description (latest instance wins
        for labs, so the most recent numeric value appears). Admission/transfer
        codes are skipped because they already appear in the
        ``CLINICAL SUMMARY`` block. Typically ~3-5× shorter than ``render()``.
        """
        parquet_glob = str(Path(meds_cohort_dir) / "data" / "*" / "*.parquet")
        lf = pl.scan_parquet(parquet_glob).filter(
            (pl.col("subject_id") == int(subject_id)) & (pl.col("time") <= prediction_time)
        )
        available = set(lf.collect_schema().names())
        select_cols = ["time", "code"]
        if "numeric_value" in available:
            select_cols.append("numeric_value")
        events = lf.select(select_cols).sort("time").collect()

        # One dict per bucket: description -> (rendered_line, last_seen_time).
        # Later occurrences overwrite earlier ones so the most recent lab
        # value wins; insertion order preserves chronology.
        diagnoses: dict[str, str] = {}
        medications: dict[str, str] = {}
        procedures: dict[str, str] = {}
        labs: dict[str, str] = {}

        for row in events.iter_rows(named=True):
            code = row["code"]
            if code is None:
                continue
            # Skip codes already summarized in the CLINICAL SUMMARY block.
            if (
                code.startswith("HOSPITAL_ADMISSION")
                or code.startswith("ICU_ADMISSION")
                or code.startswith("TRANSFER_TO")
            ):
                continue

            desc = self._code_to_description.get(code) if self.include_description else None
            if not desc:
                continue
            desc = " / ".join(p.strip() for p in str(desc).splitlines() if p.strip())

            head = code.split("//", 1)[0]
            if head == "DIAGNOSIS":
                diagnoses.pop(desc, None)
                diagnoses[desc] = f"- {desc}"
            elif head in ("MEDICATION", "INFUSION"):
                medications.pop(desc, None)
                medications[desc] = f"- {desc}"
            elif head == "PROCEDURE":
                action = _procedure_action(code)
                suffix = f" ({action})" if action is not None else ""
                key = desc + suffix
                procedures.pop(key, None)
                procedures[key] = f"- {desc}{suffix}"
            elif head == "LAB":
                clean_desc = _clean_lab_description(desc)
                unit = _unit_from_code(code)
                value = row.get("numeric_value") if "numeric_value" in row else None
                value_part = ""
                if value is not None and isinstance(value, (int, float)) and not (
                    isinstance(value, float) and value != value
                ):
                    formatted = _format_numeric(float(value))
                    if formatted:
                        value_part = f" {formatted}"
                unit_part = f" {unit}" if unit else ""
                labs.pop(desc, None)
                labs[desc] = f"- {clean_desc}:{value_part}{unit_part}".rstrip(":")
            # All other code types (SUBJECT_FLUID_OUTPUT, MEDS_BIRTH, etc.)
            # are dropped — demographics/admissions already covered elsewhere.

        def _tail(d: dict[str, str], n: int) -> list[str]:
            items = list(d.values())
            return items[-n:] if n > 0 else items

        sections: list[tuple[str, list[str]]] = [
            ("Diagnoses (from history)", _tail(diagnoses, max_diagnoses)),
            ("Active medications (recent)", _tail(medications, max_medications)),
            ("Recent procedures", _tail(procedures, max_procedures)),
            ("Recent labs (most recent value)", _tail(labs, max_labs)),
        ]

        blocks: list[str] = []
        narrative = _render_patient_narrative(
            demographics, prediction_time, clinical_summary
        )
        if narrative is not None:
            blocks.append(narrative)
        for title, items in sections:
            if not items:
                continue
            blocks.append(f"{title}:")
            blocks.extend(items)
        return "\n".join(blocks)


def _resolve_doc_row(
    doc_id: Any,
    *,
    doc_id_to_row: dict[Any, int] | None,
    n_rows: int,
) -> int | None:
    """Resolve a doc_id to a retrieval_ds row index, or ``None`` if unresolvable.

    Accepts both the case where the retriever emits real doc-id values that
    key into ``doc_id_to_row``, and the case where the retriever emits raw
    row indices (the retrieval_ds's own ``doc_ids`` column may be strings or
    absent entirely). Mirrors the fallback pattern in
    :func:`medrap.demographic_analysis._resolve_doc_row_index`.
    """
    if doc_id is None:
        return None
    if doc_id_to_row:
        if doc_id in doc_id_to_row:
            return int(doc_id_to_row[doc_id])
        try:
            key_int = int(doc_id)
        except (TypeError, ValueError):
            key_int = None
        if key_int is not None and key_int in doc_id_to_row:
            return int(doc_id_to_row[key_int])
    try:
        as_row = int(doc_id)
    except (TypeError, ValueError):
        return None
    if 0 <= as_row < n_rows:
        return as_row
    return None


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
        try:
            self._n_rows = len(retrieval_ds)
        except TypeError:
            self._n_rows = 0

    def _doc_text(self, doc_id: int) -> str:
        row = _resolve_doc_row(
            doc_id, doc_id_to_row=self.doc_id_to_row, n_rows=self._n_rows
        )
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
            f"<task>\n{self.task_description}\n</task>\n\n"
            f"<patient>\n{patient_timeline}\n</patient>\n\n"
            f"<document_a>\n{doc_a}\n</document_a>\n\n"
            f"<document_b>\n{doc_b}\n</document_b>\n\n"
            "Which document is more relevant for predicting this outcome for "
            "THIS patient? Apply the criteria and bias controls from the "
            "system instructions. Respond with a single JSON object only."
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
    # Columns that can be None throughout the first infer-schema window
    # (all-tie runs leave ``target_won`` null for > 100 rows). Pin their
    # dtypes explicitly so polars doesn't infer Null and reject later rows.
    schema_overrides = {
        "target_won": pl.Boolean,
        "other_rank": pl.Int64,
        "other_source_subject_id": pl.Int64,
        "confidence": pl.Float64,
        "prompt_tokens": pl.Int64,
        "completion_tokens": pl.Int64,
    }
    return pl.DataFrame(rows, schema_overrides=schema_overrides)


def _classify_invalid_row(winner_position: str | None, rationale: str | None) -> str:
    """Classify a verdict row by how (or whether) it failed to yield a winner.

    Returns one of: ``"valid"``, ``"tie"``, ``"api_error"``, ``"parse_error"``,
    ``"client_init_error"``, ``"other_invalid"``.

    Classification is driven by :class:`OpenAIJudge`'s rationale prefixes:
    ``"openai client init failed:"``, ``"api error:"``, ``"parse error:"``.
    """
    if winner_position in ("A", "B"):
        return "valid"
    if winner_position == "tie":
        return "tie"
    rat = (rationale or "").strip().lower()
    if rat.startswith("openai client init failed"):
        return "client_init_error"
    if rat.startswith("api error"):
        return "api_error"
    if rat.startswith("parse error"):
        return "parse_error"
    return "other_invalid"


def summarize_winrates(
    df: pl.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
    ci_level: float = 0.95,
    invalid_policy: Literal["drop", "count_as_loss", "half_credit_ties"] = "drop",
) -> pl.DataFrame:
    """Compute per-family ``target_preferred_rate`` with patient-cluster bootstrap CI.

    Algorithm:

    1. Count invalid verdicts (``target_won is None``) — surfaced as ``n_invalid``,
       plus a labeled split into ``n_ties``, ``n_api_errors``, ``n_parse_errors``,
       ``n_client_init_errors``, ``n_other_invalid`` (sum equals ``n_invalid``).
    2. Apply ``invalid_policy``:
       - ``"drop"`` removes invalid rows from numerator and denominator —
         rate = ``wins / (wins + losses)``, conditional on the judge picking a side.
       - ``"count_as_loss"`` converts invalid ``target_won`` to ``False`` —
         rate = ``wins / n_pairs``, treating ties as losses.
       - ``"half_credit_ties"`` casts ``target_won`` to float and fills invalid
         rows with ``0.5`` — rate = ``(wins + 0.5*invalid) / n_pairs``,
         the expected score under a ``{win=1, tie=0.5, loss=0}`` scoring rule
         (chess-Elo style). Note that this folds API/parse errors into the
         half-credit bin alongside true ties; restrict the input to non-error
         rows if that is undesirable.
    3. Within-patient averaging first — ``per_patient = mean(target_won)`` per
       ``(family, anchor_subject_id)`` — so a patient's pairs count once.
    4. Point estimate = ``mean(per_patient)`` across patients in the family.
    5. Patient-cluster bootstrap: resample ``N`` patients with replacement from
       the per-patient means, ``n_bootstrap`` times; SE = ``np.std(replicates, ddof=1)``;
       CI via percentile method at ``ci_level``. The bootstrap is non-parametric
       and stays valid whether per-patient values are in ``{0, 1}`` or
       ``{0, 0.5, 1}``.
    """
    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci_level) / 2.0

    has_winner = "winner_position" in df.columns
    has_rationale = "rationale" in df.columns

    results: list[dict[str, Any]] = []
    for family in sorted(df["family"].unique().to_list()):
        fam_df = df.filter(pl.col("family") == family)
        n_pairs = fam_df.height
        n_invalid = fam_df.filter(pl.col("target_won").is_null()).height

        invalid_rows = fam_df.filter(pl.col("target_won").is_null())
        winners = (
            invalid_rows["winner_position"].to_list() if has_winner else [None] * invalid_rows.height
        )
        rationales = (
            invalid_rows["rationale"].to_list() if has_rationale else [None] * invalid_rows.height
        )
        counts = {
            "tie": 0,
            "api_error": 0,
            "parse_error": 0,
            "client_init_error": 0,
            "other_invalid": 0,
        }
        for w, r in zip(winners, rationales):
            label = _classify_invalid_row(w, r)
            if label == "valid":
                # Shouldn't happen (target_won is null only when winner is tie/invalid),
                # but guard for schema variations: treat as other_invalid.
                counts["other_invalid"] += 1
            else:
                counts[label] += 1

        if invalid_policy == "drop":
            working = fam_df.filter(pl.col("target_won").is_not_null())
        elif invalid_policy == "count_as_loss":
            working = fam_df.with_columns(pl.col("target_won").fill_null(False))
        else:  # half_credit_ties
            working = fam_df.with_columns(
                pl.col("target_won").cast(pl.Float64).fill_null(0.5)
            )

        if working.height == 0:
            results.append(
                {
                    "family": family,
                    "n_patients": 0,
                    "n_pairs": n_pairs,
                    "n_invalid": n_invalid,
                    "n_ties": counts["tie"],
                    "n_api_errors": counts["api_error"],
                    "n_parse_errors": counts["parse_error"],
                    "n_client_init_errors": counts["client_init_error"],
                    "n_other_invalid": counts["other_invalid"],
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
                "n_ties": counts["tie"],
                "n_api_errors": counts["api_error"],
                "n_parse_errors": counts["parse_error"],
                "n_client_init_errors": counts["client_init_error"],
                "n_other_invalid": counts["other_invalid"],
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
    clinical_summaries_by_subject_id: dict[int, dict[str, Any]] | None = None,
    families: Sequence[str] = ("F1", "F2", "F3", "F4"),
) -> pl.DataFrame:
    """One row per sampled patient with per-family outcomes and rich metadata.

    The ``timeline_renderer`` argument is retained for API compatibility; the
    caller is expected to pre-render timelines and pass them via
    ``timelines_by_subject_id`` when it has access to the MEDS cohort dir.

    ``clinical_summaries_by_subject_id`` (output of
    :func:`compute_patient_clinical_summary`) adds healthcare-utilization
    counts + chronic-condition flags as extra columns on each row.
    """
    del timeline_renderer  # kept in the signature per plan; rendering is caller's job
    timelines = timelines_by_subject_id or {}
    summaries = clinical_summaries_by_subject_id or {}

    ds_columns: set[str] = set(getattr(retrieval_ds, "column_names", []) or [])
    available_meta = [c for c in doc_metadata_columns if c in ds_columns]
    try:
        n_rows = len(retrieval_ds)
    except TypeError:
        n_rows = 0

    def _doc_fields(doc_id: int | None) -> dict[str, Any]:
        out: dict[str, Any] = {"text_preview": None, "meta": dict.fromkeys(available_meta)}
        row = _resolve_doc_row(doc_id, doc_id_to_row=doc_id_to_row, n_rows=n_rows)
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

        summary = summaries.get(anchor_sid)
        if summary is not None:
            row["n_hospital_admissions"] = int(summary.get("n_hospital_admissions", 0))
            row["n_icu_admissions"] = int(summary.get("n_icu_admissions", 0))
            row["n_ed_visits"] = int(summary.get("n_ed_visits", 0))
            conds = summary.get("chronic_conditions") or []
            row["chronic_conditions"] = ", ".join(conds) if conds else ""
            row["chronic_condition_count"] = int(summary.get("chronic_condition_count", len(conds)))
        else:
            row["n_hospital_admissions"] = None
            row["n_icu_admissions"] = None
            row["n_ed_visits"] = None
            row["chronic_conditions"] = None
            row["chronic_condition_count"] = None

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
    try:
        n_rows = len(retrieval_ds)
    except TypeError:
        n_rows = 0

    def _doc_fields(doc_id: int | None) -> dict[str, Any]:
        out: dict[str, Any] = {"text": "", "meta": {c: None for c in available_meta}}
        row = _resolve_doc_row(doc_id, doc_id_to_row=doc_id_to_row, n_rows=n_rows)
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

    with xlsxwriter.Workbook(str(path), {"nan_inf_to_errors": True}) as wb:
        for sheet_name, frame in sheets:
            frame.write_excel(workbook=wb, worksheet=sheet_name)
