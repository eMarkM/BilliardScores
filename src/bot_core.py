"""Core logic for nilpoolbot.

This module is intentionally free of Telegram/OpenAI so it can be unit-tested.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CSV_FIELDS = [
    "player_num",
    "side",
    "player",
    "game1",
    "game2",
    "game3",
    "game4",
    "game5",
    "game6",
    "total",
]


def week_start(dt: datetime) -> datetime:
    """Return Monday 00:00 for the week containing dt (local time)."""
    d0 = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return d0 - timedelta(days=d0.weekday())


def load_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = [dict(row) for row in r]

    for row in rows:
        if "player_num" in row:
            row["player_num"] = int(row["player_num"])
        for k in list(row.keys()):
            if k.startswith("game") or k == "total":
                row[k] = int(row[k])
    return rows


def write_csv_rows(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def warnings_for_rows(rows: List[Dict[str, Any]]) -> str:
    warns: List[str] = []
    for row in rows:
        s = sum(int(row[g]) for g in ["game1", "game2", "game3", "game4", "game5", "game6"])
        if s != int(row["total"]):
            pn = row.get("player_num", "?")
            warns.append(
                f"Total mismatch for P{pn} {row['side']}:{row['player']}: games sum={s} total={row['total']}"
            )
    return "\n".join(warns)


ALLOWED_GAME_SCORES = set(range(0, 8)) | {10}


def is_valid_game_score(value: int) -> bool:
    return int(value) in ALLOWED_GAME_SCORES


def plausibility_check(
    rows: List[Dict[str, Any]],
    teams_count: int,
    home_team: Optional[int],
    visiting_team: Optional[int],
) -> Tuple[bool, List[str]]:
    """Basic guard rails to avoid accepting random photos."""

    problems: List[str] = []

    if len(rows) != 6:
        problems.append(f"Expected 6 player rows, got {len(rows)}")
        return False, problems

    if home_team is None or visiting_team is None:
        problems.append("Could not read Home/Visiting team numbers")
    else:
        if not (1 <= home_team <= teams_count):
            problems.append(f"Home team {home_team} out of range 1..{teams_count}")
        if not (1 <= visiting_team <= teams_count):
            problems.append(f"Visiting team {visiting_team} out of range 1..{teams_count}")
        if home_team == visiting_team:
            problems.append("Home and Visiting team numbers are identical")

    name_ok = 0
    for r in rows:
        name = (r.get("player") or "").strip()
        if len(name) >= 2 and any(c.isalpha() for c in name):
            name_ok += 1
    if name_ok < 4:
        problems.append(f"Too few readable player names ({name_ok}/6)")

    mismatch = 0
    for r in rows:
        s = sum(int(r[g]) for g in ["game1", "game2", "game3", "game4", "game5", "game6"])
        if s != int(r["total"]):
            mismatch += 1
    if mismatch > 2:
        problems.append(f"Too many total mismatches ({mismatch}/6)")

    distinct_scores = set()
    invalid = 0
    for r in rows:
        for g in ["game1", "game2", "game3", "game4", "game5", "game6"]:
            v = int(r[g])
            if not is_valid_game_score(v):
                invalid += 1
            distinct_scores.add(v)
    if invalid:
        problems.append(f"Found {invalid} invalid game values (valid: 0-7 or 10)")
    if len(distinct_scores) <= 1:
        problems.append("All game values appear identical (likely not a scoresheet)")

    return (len(problems) == 0), problems


def apply_fixname(rows: List[Dict[str, Any]], player_num: int, new_name: str) -> Tuple[bool, str]:
    matches = [r for r in rows if int(r.get("player_num", -1)) == player_num]
    if not matches:
        return False, f"Couldn't find player number {player_num}"
    for r in matches:
        r["player"] = new_name
    return True, ""


def apply_fixscore(rows: List[Dict[str, Any]], player_num: int, game_num: int, value: int) -> Tuple[bool, str]:
    if not (1 <= player_num <= 6):
        return False, "player_num must be 1-6"

    if game_num == 0:
        field = "total"
    elif 1 <= game_num <= 6:
        field = f"game{game_num}"
        if not is_valid_game_score(value):
            return False, "Game values must be 0-7 or 10"
    else:
        return False, "game_num must be 1-6, or 0 for total"

    matches = [r for r in rows if int(r.get("player_num", -1)) == player_num]
    if not matches:
        return False, f"Couldn't find player number {player_num}"

    for r in matches:
        r[field] = value
    return True, ""


def build_processed_caption(
    *,
    image_filename: str,
    message_id: Optional[str],
    home_team: Optional[int],
    warn_text: Optional[str],
) -> str:
    """Human-friendly caption for the CSV reply.

    Keep this in bot_core so it stays easy to unit-test.
    """

    if home_team is None:
        header = "Scoresheet has been successfully processed."
    else:
        header = f"Team {home_team} has been successfully processed."

    body = (
        f"{header}\n"
        "Please compare to the physical sheet and send /confirm if correct.\n"
        "Use /fixname and /fixscore to fix errors.\n\n"
        f"Processed photo: {image_filename}"
    )

    if message_id:
        body += f"\n[message_id: {message_id}]"

    if warn_text:
        body += f"\n\nWarnings:\n{warn_text}"

    return body
