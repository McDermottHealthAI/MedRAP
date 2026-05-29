"""Tests for the multitask additions to ``scripts/run_llm_judge.py``.

The script doesn't ship a test file today; this is a focused new file that
covers the two helpers added for multitask support:

- ``_select_target_task``: 2-D-targets-to-1-D slice + NaN row filter.
- ``_auto_task_description``: natural-language description per MEDS code.

The natural-language guardrail (no raw ``LAB//<itemid>//<unit>``-style
codes leaking into prompts) is the core safety property; we assert it
against the real ``data/mt_labels/top25_7d/code_index.json`` when present,
and across a hand-crafted set of edge-case codes otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Importing run_llm_judge runs the module-level sys.path mutation that pulls in
# medrap and extract_and_visualize helpers. Keep the import isolated to this
# test file so the rest of the suite isn't affected.
import run_llm_judge  # noqa: E402

_select_target_task = run_llm_judge._select_target_task
_auto_task_description = run_llm_judge._auto_task_description
_resolve_task_mode = run_llm_judge._resolve_task_mode
_families_require_both_classes = run_llm_judge._families_require_both_classes


# ---------------------------------------------------------------------------
# _families_require_both_classes
# ---------------------------------------------------------------------------


def test_families_require_both_classes_F1_only_is_false() -> None:
    """F1 (random corpus doc) ignores labels, so a dummy-zeros label vector should not trip the both-classes-
    present guardrail in ``_run_one_task``."""
    assert _families_require_both_classes(("F1",)) is False


def test_families_require_both_classes_F2_only_is_false() -> None:
    """F2 (same-patient lower rank) also ignores labels."""
    assert _families_require_both_classes(("F2",)) is False


def test_families_require_both_classes_F3_is_true() -> None:
    assert _families_require_both_classes(("F3",)) is True


def test_families_require_both_classes_F4_is_true() -> None:
    assert _families_require_both_classes(("F4",)) is True


def test_families_require_both_classes_mixed_is_true_if_any_label_dependent() -> None:
    assert _families_require_both_classes(("F1", "F3")) is True
    assert _families_require_both_classes(("F1", "F2", "F3", "F4")) is True


def test_families_require_both_classes_empty_is_false() -> None:
    assert _families_require_both_classes(()) is False


# ---------------------------------------------------------------------------
# _resolve_task_mode
# ---------------------------------------------------------------------------


def test_resolve_task_mode_auto_maps_2d_targets_to_multitask() -> None:
    """Default behavior on a multitask checkpoint is the per-task 25-task sweep.

    The single overall-pool sweep is reachable explicitly via
    ``--task_mode overall``.
    """
    two_d = np.zeros((4, 25), dtype=float)
    assert _resolve_task_mode("auto", two_d) == "multitask"


def test_resolve_task_mode_auto_maps_1d_targets_to_binary() -> None:
    one_d = np.zeros(4, dtype=float)
    assert _resolve_task_mode("auto", one_d) == "binary"


def test_resolve_task_mode_accepts_explicit_overall() -> None:
    two_d = np.zeros((4, 25), dtype=float)
    assert _resolve_task_mode("overall", two_d) == "overall"


def test_resolve_task_mode_preserves_explicit_multitask() -> None:
    """Soft change: explicit ``--task_mode multitask`` still reaches the
    25-task sweep."""
    two_d = np.zeros((4, 25), dtype=float)
    assert _resolve_task_mode("multitask", two_d) == "multitask"


def test_resolve_task_mode_preserves_explicit_binary() -> None:
    one_d = np.zeros(4, dtype=float)
    assert _resolve_task_mode("binary", one_d) == "binary"


def test_resolve_task_mode_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown task_mode"):
        _resolve_task_mode("bogus", np.zeros(4, dtype=float))


# ---------------------------------------------------------------------------
# _select_target_task
# ---------------------------------------------------------------------------


def test_select_target_task_filters_nan_and_returns_aligned_arrays() -> None:
    """NaN-labeled rows for the chosen task are dropped; valid_indices align."""
    nan = float("nan")
    raw = np.array(
        [
            [1.0, 0.0, nan],  # row 0
            [0.0, nan, 1.0],  # row 1
            [1.0, 1.0, 1.0],  # row 2
            [nan, 1.0, 0.0],  # row 3
            [0.0, 0.0, nan],  # row 4
        ],
        dtype=float,
    )
    # Task 1 column: [0.0, nan, 1.0, 1.0, 0.0] → valid rows {0, 2, 3, 4}.
    labels, valid_indices, meta = _select_target_task(
        raw,
        target_task=1,
        task_codes={1: "MEDICATION//START//Heparin"},
        lab_lookup=None,
        horizon_days=7.0,
        anchor_offset_hours=24.0,
    )
    assert labels.tolist() == [0, 1, 1, 0]
    assert labels.dtype.kind == "i"
    assert valid_indices.tolist() == [0, 2, 3, 4]
    assert meta["target_task"] == 1
    assert meta["n_valid"] == 4
    assert meta["n_pos"] == 2
    assert meta["n_neg"] == 2
    assert meta["task_code"] == "MEDICATION//START//Heparin"
    # The label is humanized — no // leak.
    assert "//" not in meta["task_label"]
    assert "//" not in meta["task_description"]
    assert "Heparin" in meta["task_description"]


def test_select_target_task_handles_no_task_code() -> None:
    """When ``task_codes`` is None or missing the index, we still return a plain numbered label rather than
    crashing."""
    raw = np.array([[1.0], [0.0]], dtype=float)
    labels, valid_indices, meta = _select_target_task(
        raw,
        target_task=0,
        task_codes=None,
        lab_lookup=None,
        horizon_days=7.0,
        anchor_offset_hours=24.0,
    )
    assert meta["task_code"] is None
    assert meta["task_label"] == "task 0"
    assert "//" not in meta["task_description"]
    assert valid_indices.tolist() == [0, 1]
    assert labels.tolist() == [1, 0]


def test_select_target_task_rejects_out_of_range_index() -> None:
    raw = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    with pytest.raises(ValueError, match="out of range"):
        _select_target_task(
            raw,
            target_task=5,
            task_codes=None,
            lab_lookup=None,
            horizon_days=7.0,
            anchor_offset_hours=24.0,
        )


def test_select_target_task_rejects_1d_targets() -> None:
    """1-D targets are the binary path; the function must refuse them."""
    raw = np.array([1.0, 0.0, 1.0], dtype=float)
    with pytest.raises(ValueError, match="2-D"):
        _select_target_task(
            raw,
            target_task=0,
            task_codes=None,
            lab_lookup=None,
            horizon_days=7.0,
            anchor_offset_hours=24.0,
        )


# ---------------------------------------------------------------------------
# _auto_task_description — natural-language guardrail
# ---------------------------------------------------------------------------


def _natural_language_invariants(
    description: str, *, horizon_days: float, anchor_offset_hours: float
) -> None:
    """Properties every auto-generated task description must satisfy."""
    # Hard requirement: no raw `//`-delimited MEDS codes leak into the prompt.
    assert "//" not in description, f"raw code leaked: {description!r}"
    # Mention the horizon and anchor; without these, the LLM has no time
    # frame to reason against.
    assert f"{horizon_days:g} days" in description, description
    assert f"{anchor_offset_hours:g} h" in description, description
    # A description should be a real sentence ending with a period.
    assert description.strip().endswith("."), description


@pytest.mark.parametrize(
    "code, expected_phrases",
    [
        # MEDICATION//START//<drug>
        ("MEDICATION//START//Heparin", ["Heparin", "started"]),
        ("MEDICATION//STOP//Heparin", ["Heparin", "stopped"]),
        ("MEDICATION//START//Sodium Chloride 0.9%  Flush", ["Sodium Chloride", "started"]),
        # MEDICATION//<drug>//Administered
        ("MEDICATION//Acetaminophen//Administered", ["Acetaminophen", "administered"]),
        # MEDICATION with UNK drug
        ("MEDICATION//START//UNK", ["medication", "started"]),
        # TRANSFER_TO//discharge → discharged from the hospital
        ("TRANSFER_TO//discharge//UNKNOWN", ["discharged", "hospital"]),
        # ED_OUT
        ("ED_OUT", ["ED departure"]),
    ],
)
def test_auto_task_description_known_prefixes(code: str, expected_phrases: list[str]) -> None:
    description = _auto_task_description(code, lab_lookup=None, horizon_days=7.0, anchor_offset_hours=24.0)
    _natural_language_invariants(description, horizon_days=7.0, anchor_offset_hours=24.0)
    for phrase in expected_phrases:
        assert phrase.lower() in description.lower(), (phrase, description)


def test_auto_task_description_lab_with_lookup() -> None:
    """When d_labitems lookup contains the itemid, the test name appears verbatim."""
    lab_lookup = {51484: ("Ketone", "Urine")}
    description = _auto_task_description(
        "LAB//51484//mg/dL",
        lab_lookup=lab_lookup,
        horizon_days=7.0,
        anchor_offset_hours=24.0,
    )
    _natural_language_invariants(description, horizon_days=7.0, anchor_offset_hours=24.0)
    assert "Ketone" in description
    assert "Urine" in description


def test_auto_task_description_lab_without_lookup_does_not_leak_code() -> None:
    """Even without a d_labitems lookup, the description must not leak //."""
    description = _auto_task_description(
        "LAB//51484//mg/dL",
        lab_lookup=None,
        horizon_days=7.0,
        anchor_offset_hours=24.0,
    )
    _natural_language_invariants(description, horizon_days=7.0, anchor_offset_hours=24.0)
    assert "51484" in description  # itemid is acceptable; raw code is not
    assert "lab test" in description.lower()


def test_auto_task_description_unknown_prefix_uses_generic_phrase() -> None:
    """Unknown prefix must still produce a natural-language sentence (no // leak)."""
    description = _auto_task_description(
        "VITAL//heart_rate//bpm",
        lab_lookup=None,
        horizon_days=7.0,
        anchor_offset_hours=24.0,
    )
    _natural_language_invariants(description, horizon_days=7.0, anchor_offset_hours=24.0)


def test_auto_task_description_for_all_25_real_codes_is_natural_language() -> None:
    """Strongest version of the guardrail: every real task code in
    ``data/mt_labels/top25_7d/code_index.json`` produces a natural-language
    description with no raw `//` codes leaking. Skipped if the file isn't
    present (e.g., on a fresh checkout without label artifacts)."""
    code_index_path = _REPO_ROOT / "data" / "mt_labels" / "top25_7d" / "code_index.json"
    if not code_index_path.is_file():
        pytest.skip(f"{code_index_path} not present in this checkout")

    codes = json.loads(code_index_path.read_text())
    # Optional d_labitems lookup — if available, descriptions get richer; if
    # not, they fall back to "with internal item id <N>".
    lab_lookup = {}
    labitems_path = Path("/groups/mm6677_gp/data/MIMIC_MEDS/raw_input/hosp/d_labitems.csv.gz")
    if labitems_path.is_file():
        from extract_and_visualize import _load_lab_label_lookup

        loaded = _load_lab_label_lookup(labitems_path)
        if loaded is not None:
            lab_lookup = loaded

    horizon_days = 7.0
    anchor_offset_hours = 24.0
    for idx_str, code in codes.items():
        idx = int(idx_str)
        description = _auto_task_description(
            code,
            lab_lookup=lab_lookup or None,
            horizon_days=horizon_days,
            anchor_offset_hours=anchor_offset_hours,
        )
        _natural_language_invariants(
            description, horizon_days=horizon_days, anchor_offset_hours=anchor_offset_hours
        )
        assert "lab//" not in description.lower(), (idx, code, description)
        assert "medication//" not in description.lower(), (idx, code, description)
        assert "transfer_to//" not in description.lower(), (idx, code, description)
