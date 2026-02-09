from datetime import datetime
from pathlib import Path

from bot_core import (
    week_start,
    warnings_for_rows,
    plausibility_check,
    apply_fixname,
    apply_fixscore,
    write_csv_rows,
    load_csv_rows,
)


def sample_rows():
    # Use varied scores so plausibility_check doesn't flag "all identical".
    return [
        {"player_num": 1, "side": "home", "player": "Ann", "game1": 1, "game2": 2, "game3": 1, "game4": 1, "game5": 1, "game6": 0, "total": 6},
        {"player_num": 2, "side": "home", "player": "Bob", "game1": 2, "game2": 1, "game3": 1, "game4": 0, "game5": 1, "game6": 1, "total": 6},
        {"player_num": 3, "side": "home", "player": "Cal", "game1": 1, "game2": 1, "game3": 2, "game4": 1, "game5": 1, "game6": 0, "total": 6},
        {"player_num": 4, "side": "visiting", "player": "Dee", "game1": 1, "game2": 1, "game3": 1, "game4": 2, "game5": 0, "game6": 1, "total": 6},
        {"player_num": 5, "side": "visiting", "player": "Eli", "game1": 0, "game2": 1, "game3": 1, "game4": 1, "game5": 2, "game6": 1, "total": 6},
        {"player_num": 6, "side": "visiting", "player": "Fay", "game1": 1, "game2": 0, "game3": 1, "game4": 1, "game5": 1, "game6": 2, "total": 6},
    ]


def test_week_start_monday():
    dt = datetime(2026, 2, 3, 10, 0, 0)  # Tue
    ws = week_start(dt)
    assert ws.weekday() == 0
    assert ws.hour == 0 and ws.minute == 0


def test_warnings_total_mismatch():
    rows = sample_rows()
    rows[0]["total"] = 5
    w = warnings_for_rows(rows)
    assert "Total mismatch" in w
    assert "P1" in w


def test_plausibility_accepts_good():
    ok, problems = plausibility_check(sample_rows(), teams_count=14, home_team=5, visiting_team=2)
    assert ok
    assert problems == []


def test_plausibility_rejects_impossible_score_9():
    rows = sample_rows()
    rows[0]["game1"] = 9
    ok, problems = plausibility_check(rows, teams_count=14, home_team=5, visiting_team=2)
    assert not ok
    assert any("invalid game" in p.lower() for p in problems)


def test_plausibility_rejects_missing_teams():
    ok, problems = plausibility_check(sample_rows(), teams_count=14, home_team=None, visiting_team=None)
    assert not ok
    assert any("Could not read" in p for p in problems)


def test_apply_fixname():
    rows = sample_rows()
    ok, err = apply_fixname(rows, 6, "Anthony")
    assert ok
    assert rows[-1]["player"] == "Anthony"


def test_apply_fixscore_game_updates_total():
    rows = sample_rows()
    before_total = rows[3]["total"]

    ok, err = apply_fixscore(rows, 4, 2, 10)
    assert ok
    assert rows[3]["game2"] == 10
    assert rows[3]["total"] == before_total + 9  # game2 changed from 1 -> 10


def test_apply_fixscore_rejects_9():
    rows = sample_rows()
    ok, err = apply_fixscore(rows, 4, 2, 9)
    assert not ok
    assert "0-7 or 10" in err


def test_apply_fixscore_total():
    rows = sample_rows()
    ok, err = apply_fixscore(rows, 4, 0, 99)
    assert ok
    assert rows[3]["total"] == 99


def test_csv_roundtrip(tmp_path: Path):
    p = tmp_path / "x.csv"
    rows = sample_rows()
    write_csv_rows(p, rows)
    rows2 = load_csv_rows(p)
    assert rows2 == rows
