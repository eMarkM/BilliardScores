from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from extract_nil_sample_to_csv import extract_rows_by_cropping, normalize_rows
from vision_client import CassetteVisionClient, OpenAIVisionClient


DATA = Path(__file__).resolve().parent / "data"
CASE_IMAGE = DATA / "testscoresheet.jpg"
CASE_EXPECTED = DATA / "testscoresheet.json"
CASSETTE_DIR = Path(__file__).resolve().parent / "cassettes" / "testscoresheet"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "_artifacts" / "testscoresheet"


def recompute_totals(rows: list[dict]) -> None:
    for r in rows:
        r["total"] = sum(int(r[f"game{i}"]) for i in range(1, 7))


def assert_scores_in_range(rows: list[dict]) -> None:
    for r in rows:
        for i in range(1, 7):
            v = int(r[f"game{i}"])
            assert 0 <= v <= 10, f"Out-of-range game{i}={v} for P{r.get('player_num')} {r.get('side')}"


def compare_rows(expected: list[dict], actual: list[dict]) -> tuple[list[str], list[str]]:
    """Return (errors, player_name_warnings)."""

    exp_by = {(int(r["player_num"]), r["side"]): r for r in expected}
    act_by = {(int(r["player_num"]), r["side"]): r for r in actual}

    errors: list[str] = []
    warns: list[str] = []

    for key, er in exp_by.items():
        ar = act_by.get(key)
        if not ar:
            errors.append(f"Missing row {key} in actual")
            continue

        # Scores must match exactly
        for i in range(1, 7):
            k = f"game{i}"
            if int(er[k]) != int(ar[k]):
                errors.append(f"{key} {k}: expected {er[k]} got {ar[k]}")

        # Player names: warn only (handwriting/OCR)
        ep = (er.get("player") or "").strip()
        ap = (ar.get("player") or "").strip()
        if ep and ap and ep.lower() != ap.lower():
            warns.append(f"{key} player: expected '{ep}' got '{ap}'")

    # Ensure no unexpected keys missing
    if len(actual) != len(expected):
        errors.append(f"Row count mismatch: expected {len(expected)} got {len(actual)}")

    return errors, warns


@pytest.mark.integration
def test_testscoresheet_replay_or_record():
    if not CASE_IMAGE.exists():
        pytest.skip(f"Missing {CASE_IMAGE}")
    expected = json.loads(CASE_EXPECTED.read_text(encoding="utf-8"))

    mode = os.getenv("VISION_TEST_MODE", "replay")  # replay|record|live

    if mode == "live":
        client = OpenAIVisionClient()
    else:
        live = OpenAIVisionClient() if mode == "record" else None
        client = CassetteVisionClient(CASSETTE_DIR, mode=mode, live=live)

    # Always generate artifacts for inspection.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        extracted = extract_rows_by_cropping(
            CASE_IMAGE,
            model=os.getenv("BILLIARDSCORES_MODEL", "gpt-4o"),
            client=client,
            debug_dir=ARTIFACTS_DIR,
        )
    except FileNotFoundError as e:
        if mode == "replay":
            pytest.skip(f"{e} (set VISION_TEST_MODE=record to create cassettes)")
        raise
    rows = normalize_rows(CASE_IMAGE.name, extracted)

    assert_scores_in_range(rows)
    recompute_totals(rows)

    errors, warns = compare_rows(expected, rows)

    if warns:
        # Surface as test output without failing.
        print("\n".join(["PLAYER WARNINGS:"] + warns))

    if errors:
        # Keep artifacts and provide pointer.
        raise AssertionError("\n".join(errors) + f"\nArtifacts: {ARTIFACTS_DIR}")
