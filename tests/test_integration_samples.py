from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from extract_nil_sample_to_csv import extract_rows_by_cropping, normalize_rows
from vision_client import CassetteVisionClient, OpenAIVisionClient


DATA = Path(__file__).resolve().parent / "data"
CASSETTES = Path(__file__).resolve().parent / "cassettes"
ARTIFACTS = Path(__file__).resolve().parent / "_artifacts"


CASES = [
    {
        "name": "Team11vTeam2",
        "image": DATA / "scoresheet-Team11vTeam2.jpg",
        "expected": DATA / "expected-json-Team11vTeam2.json",
        "cassette_dir": CASSETTES / "scoresheet-Team11vTeam2",
        "artifacts_dir": ARTIFACTS / "scoresheet-Team11vTeam2",
    },
    {
        "name": "Team14vTeam9",
        "image": DATA / "scoresheet-Team14vTeam9.jpg",
        "expected": DATA / "expected-json-Team14vTeam9.json",
        "cassette_dir": CASSETTES / "scoresheet-Team14vTeam9",
        "artifacts_dir": ARTIFACTS / "scoresheet-Team14vTeam9",
    },
]


def recompute_totals(rows: list[dict]) -> None:
    for r in rows:
        r["total"] = sum(int(r[f"game{i}"]) for i in range(1, 7))


def assert_row_shape(rows: list[dict]) -> None:
    assert len(rows) == 6, f"Expected 6 rows, got {len(rows)}"

    keys = [(int(r.get("player_num")), str(r.get("side"))) for r in rows]
    if len(set(keys)) != 6:
        raise AssertionError(f"Duplicate (player_num, side) detected: {keys}")

    players = [str(r.get("player") or "").strip().lower() for r in rows]
    assert all(players), f"Blank player name detected: {players}"

    dups = {p for p in players if players.count(p) > 1}
    assert not dups, f"Repeated player names detected (normalized): {sorted(dups)}"


def compare_scores_with_tolerance(
    expected: list[dict],
    actual: list[dict],
    *,
    per_cell_tolerance: int = 2,
) -> list[str]:
    exp_by = {(int(r["player_num"]), r["side"]): r for r in expected}
    act_by = {(int(r["player_num"]), r["side"]): r for r in actual}

    errors: list[str] = []

    for key, er in exp_by.items():
        ar = act_by.get(key)
        if not ar:
            errors.append(f"Missing row {key} in actual")
            continue

        for i in range(1, 7):
            k = f"game{i}"
            ev = int(er[k])
            av = int(ar[k])
            if abs(ev - av) > per_cell_tolerance:
                errors.append(f"{key} {k}: expected {ev} got {av} (|diff|={abs(ev-av)})")

    if len(actual) != len(expected):
        errors.append(f"Row count mismatch: expected {len(expected)} got {len(actual)}")

    return errors


@pytest.mark.integration
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_sample_scoresheets_replay_or_record(case: dict):
    image: Path = case["image"]
    expected_path: Path = case["expected"]

    if not image.exists():
        pytest.skip(f"Missing {image}")

    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    mode = os.getenv("VISION_TEST_MODE", "replay")  # replay|record|live

    if mode == "live":
        client = OpenAIVisionClient()
    else:
        live = OpenAIVisionClient() if mode == "record" else None
        client = CassetteVisionClient(case["cassette_dir"], mode=mode, live=live)

    artifacts_dir: Path = case["artifacts_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    extracted = extract_rows_by_cropping(
        image,
        model=os.getenv("BILLIARDSCORES_MODEL", "gpt-4o"),
        client=client,
        debug_dir=artifacts_dir,
    )
    rows = normalize_rows(image.name, extracted)

    recompute_totals(rows)
    assert_row_shape(rows)

    errors = compare_scores_with_tolerance(expected, rows, per_cell_tolerance=2)
    if errors:
        raise AssertionError("\n".join(errors) + f"\nArtifacts: {artifacts_dir}")
