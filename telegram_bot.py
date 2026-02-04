#!/usr/bin/env python3
"""Telegram bot: receive NIL scoresheet photos and return CSV.

Local-first prototype:
- Captains send a photo to the bot.
- Bot downloads the photo, runs the extractor (OpenAI vision), and replies with:
  - CSV file
  - any validation warnings

Env:
  TELEGRAM_BOT_TOKEN=...   (from @BotFather)
  OPENAI_API_KEY=...       (for vision extraction)
  BILLIARDSCORES_MODEL=... (optional, default gpt-4o-mini)

Run:
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  export TELEGRAM_BOT_TOKEN=... OPENAI_API_KEY=...
  python3 telegram_bot.py
"""

from __future__ import annotations

import csv
import logging
from logging.handlers import RotatingFileHandler
import os
import shlex
import sqlite3
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters


HERE = Path(__file__).resolve().parent
UPLOADS_DIR = HERE / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = HERE / "bot.db"
LOG_PATH = HERE / "bot.log"

logger = logging.getLogger("nilpoolbot")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    # Also log to stderr for the console
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)


HELP_TEXT = (
    "Send me a clear photo of the NIL scoresheet and I’ll reply with a CSV.\n\n"
    "Workflow:\n"
    "1) Send photo → I reply with a CSV and mark it PENDING\n"
    "2) If it looks good, reply /confirm to lock it in\n"
    "3) If not, use /fixscore or /fixname (or re-upload a clearer photo)\n\n"
    "Commands:\n"
    "- /confirm — confirm your latest pending upload for this week\n"
    "- /status — show team upload status since Monday\n"
    "- /fixscore — correct a score in your pending CSV\n"
    "- /fixname — correct a player name in your pending CSV\n"
    "- /help — show this help\n\n"
    "Your CSV includes a player number (P1..P6). Use that for fixes.\n\n"
    "Fix score format:\n"
    "  /fixscore <player_num> <game_num> <value>\n"
    "Examples:\n"
    "  /fixscore 4 2 4     (player 4, game 2 → 4)\n"
    "  /fixscore 3 4 5     (player 3, game 4 → 5)\n\n"
    "Fix name format:\n"
    "  /fixname <player_num> <new_name>\n"
    "Example:\n"
    "  /fixname 6 Anthony\n\n"
    "Tips for best results:\n"
    "- Fill the frame with the sheet\n"
    "- Avoid shadows/glare\n"
    "- Hold camera parallel to paper\n"
)


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS uploads (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL,
          week_start TEXT NOT NULL,
          chat_id INTEGER,
          chat_title TEXT,
          user_id INTEGER,
          username TEXT,
          first_name TEXT,
          last_name TEXT,
          image_path TEXT NOT NULL,
          csv_path TEXT,
          warnings TEXT,
          confirmed_at TEXT,
          home_team INTEGER,
          visiting_team INTEGER
        )
        """
    )

    # Lightweight migrations for older DBs
    for stmt in [
        "ALTER TABLE uploads ADD COLUMN confirmed_at TEXT",
        "ALTER TABLE uploads ADD COLUMN home_team INTEGER",
        "ALTER TABLE uploads ADD COLUMN visiting_team INTEGER",
    ]:
        try:
            con.execute(stmt)
            con.commit()
        except sqlite3.OperationalError:
            pass

    return con


def _week_start(dt: datetime) -> datetime:
    # Monday 00:00 local
    d0 = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return d0 - timedelta(days=d0.weekday())


def _teams_count() -> int:
    # total number of teams in the league
    try:
        return int((os.getenv("TEAMS_COUNT") or "14").strip())
    except Exception:
        return 14


def _user_label(update: Update) -> str:
    u = update.effective_user
    if not u:
        return "(unknown)"
    if u.username:
        return f"@{u.username}"
    name = " ".join([x for x in [u.first_name, u.last_name] if x])
    return name or str(u.id)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("help chat_id=%s user=%s", getattr(update.effective_chat, "id", None), _user_label(update))
    await update.message.reply_text(HELP_TEXT)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await help_cmd(update, context)


async def recent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    con = _db()
    try:
        rows = con.execute(
            "SELECT created_at, username, first_name, last_name, chat_title, image_path FROM uploads ORDER BY id DESC LIMIT 10"
        ).fetchall()
    finally:
        con.close()

    if not rows:
        await update.message.reply_text("No uploads yet.")
        return

    lines = ["Recent uploads:"]
    for created_at, username, first_name, last_name, chat_title, image_path in rows:
        who = f"@{username}" if username else " ".join([x for x in [first_name, last_name] if x])
        where = f" in {chat_title}" if chat_title else ""
        lines.append(f"- {created_at}: {who}{where} ({Path(image_path).name})")

    await update.message.reply_text("\n".join(lines))


async def pending(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    now = datetime.now()
    ws_s = _week_start(now).date().isoformat()

    con = _db()
    try:
        rows = con.execute(
            """
            SELECT created_at, id, home_team, visiting_team
            FROM uploads
            WHERE week_start = ? AND confirmed_at IS NULL
            ORDER BY id DESC
            LIMIT 50
            """,
            (ws_s,),
        ).fetchall()
    finally:
        con.close()

    if not rows:
        await update.message.reply_text(f"No pending uploads for week starting {ws_s}.")
        return

    lines = [f"Pending uploads for week starting {ws_s}:"]
    for created_at, upload_id, home_team, visiting_team in rows:
        t = ""
        if home_team is not None and visiting_team is not None:
            t = f" Team{home_team}vTeam{visiting_team}"
        lines.append(f"- #{upload_id} {created_at}:{t}")

    await update.message.reply_text("\n".join(lines))


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Team-based weekly status.

    For v1, we treat the upload as belonging to the HOME team on the sheet.
    Week resets on Monday.
    """

    now = datetime.now()
    ws_s = _week_start(now).date().isoformat()
    n_teams = _teams_count()

    # Build team -> status (None/Pending/Confirmed)
    team_status: Dict[int, str] = {i: "None" for i in range(1, n_teams + 1)}

    con = _db()
    try:
        rows = con.execute(
            """
            SELECT home_team, confirmed_at
            FROM uploads
            WHERE week_start = ? AND home_team IS NOT NULL
            ORDER BY id ASC
            """,
            (ws_s,),
        ).fetchall()
    finally:
        con.close()

    for home_team, confirmed_at in rows:
        if home_team is None:
            continue
        t = int(home_team)
        if t < 1 or t > n_teams:
            continue

        # Confirmed beats Pending beats None
        if confirmed_at:
            team_status[t] = "Confirmed"
        else:
            if team_status[t] != "Confirmed":
                team_status[t] = "Pending"

    lines = [f"Team upload status for week starting {ws_s}:"]
    for t in range(1, n_teams + 1):
        lines.append(f"- Team {t}: {team_status[t]}")

    await update.message.reply_text("\n".join(lines))


def _load_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = [dict(row) for row in r]
    # normalize ints
    for row in rows:
        if "player_num" in row:
            row["player_num"] = int(row["player_num"])
        for k in list(row.keys()):
            if k.startswith("game") or k == "total":
                row[k] = int(row[k])
    return rows


def _write_csv_rows(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ["player_num", "side", "player", "game1", "game2", "game3", "game4", "game5", "game6", "total"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _warnings_for_rows(rows: List[Dict[str, Any]]) -> str:
    warns: List[str] = []
    for row in rows:
        s = sum(int(row[g]) for g in ["game1", "game2", "game3", "game4", "game5", "game6"])
        if s != int(row["total"]):
            pn = row.get("player_num", "?")
            warns.append(
                f"Total mismatch for P{pn} {row['side']}:{row['player']}: games sum={s} total={row['total']}"
            )
    return "\n".join(warns)


def _plausibility_check(
    rows: List[Dict[str, Any]],
    teams_count: int,
    home_team: Optional[int],
    visiting_team: Optional[int],
) -> Tuple[bool, List[str]]:
    """Basic guard rails to avoid accepting random photos.

    We don't need perfection—just catch obvious non-scoresheets.
    """

    problems: List[str] = []

    if len(rows) != 6:
        problems.append(f"Expected 6 player rows, got {len(rows)}")
        return False, problems

    # Team numbers should be present and in-range for a real sheet.
    if home_team is None or visiting_team is None:
        problems.append("Could not read Home/Visiting team numbers")
    else:
        if not (1 <= home_team <= teams_count):
            problems.append(f"Home team {home_team} out of range 1..{teams_count}")
        if not (1 <= visiting_team <= teams_count):
            problems.append(f"Visiting team {visiting_team} out of range 1..{teams_count}")
        if home_team == visiting_team:
            problems.append("Home and Visiting team numbers are identical")

    # Names should look like names: at least 4 non-trivial strings.
    name_ok = 0
    for r in rows:
        name = (r.get("player") or "").strip()
        if len(name) >= 2 and any(c.isalpha() for c in name):
            name_ok += 1
    if name_ok < 4:
        problems.append(f"Too few readable player names ({name_ok}/6)")

    # Totals should match sum of games for most rows.
    mismatch = 0
    for r in rows:
        s = sum(int(r[g]) for g in ["game1", "game2", "game3", "game4", "game5", "game6"])
        if s != int(r["total"]):
            mismatch += 1
    if mismatch > 2:
        problems.append(f"Too many total mismatches ({mismatch}/6)")

    # Games should be in range and not all identical junk.
    distinct_scores = set()
    out_of_range = 0
    for r in rows:
        for g in ["game1", "game2", "game3", "game4", "game5", "game6"]:
            v = int(r[g])
            if v < 0 or v > 10:
                out_of_range += 1
            distinct_scores.add(v)
    if out_of_range:
        problems.append(f"Found {out_of_range} out-of-range game values")
    if len(distinct_scores) <= 1:
        problems.append("All game values appear identical (likely not a scoresheet)")

    ok = len(problems) == 0
    return ok, problems


def _latest_pending_upload(con: sqlite3.Connection, ws_s: str, chat_id: int, user_id: int):
    return con.execute(
        """
        SELECT id, image_path, csv_path
        FROM uploads
        WHERE week_start = ? AND chat_id = ? AND user_id = ? AND confirmed_at IS NULL
        ORDER BY id DESC
        LIMIT 1
        """,
        (ws_s, chat_id, user_id),
    ).fetchone()


async def confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg:
        return

    u = update.effective_user
    c = update.effective_chat
    if not u or not c:
        await msg.reply_text("Can’t determine user/chat.")
        return

    ws_s = _week_start(datetime.now()).date().isoformat()

    con = _db()
    try:
        row = con.execute(
            """
            SELECT id, created_at, image_path
            FROM uploads
            WHERE week_start = ? AND chat_id = ? AND user_id = ? AND confirmed_at IS NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (ws_s, c.id, u.id),
        ).fetchone()

        if not row:
            logger.info(
                "confirm_no_pending week_start=%s chat_id=%s user=%s",
                ws_s,
                c.id,
                _user_label(update),
            )
            await msg.reply_text("No pending upload found to confirm (for this week).")
            return

        upload_id, created_at, image_path = row
        con.execute(
            "UPDATE uploads SET confirmed_at = ? WHERE id = ?",
            (datetime.now().isoformat(timespec="seconds"), upload_id),
        )
        con.commit()
    finally:
        con.close()

    logger.info(
        "confirm_ok upload_id=%s week_start=%s chat_id=%s user=%s image=%s",
        upload_id,
        ws_s,
        c.id,
        _user_label(update),
        Path(image_path).name,
    )
    await msg.reply_text(
        f"Confirmed upload #{upload_id} from {created_at} ({Path(image_path).name}). Thanks."
    )


async def fixscore(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg:
        return

    u = update.effective_user
    c = update.effective_chat
    if not u or not c:
        await msg.reply_text("Can’t determine user/chat.")
        return

    # Parse with quotes support
    parts = shlex.split(msg.text or "")
    if len(parts) != 4:
        await msg.reply_text(
            "Usage: /fixscore <player_num> <game_num> <value>\n"
            "Example: /fixscore 4 2 4"
        )
        return

    _, player_num_s, game_num_s, value_s = parts

    try:
        player_num = int(player_num_s)
    except ValueError:
        await msg.reply_text("player_num must be an integer (1-6).")
        return

    try:
        game_num = int(game_num_s)
    except ValueError:
        await msg.reply_text("game_num must be an integer (1-6), or use 0 to mean TOTAL.")
        return

    try:
        value = int(value_s)
    except ValueError:
        await msg.reply_text("value must be an integer.")
        return

    if not (1 <= player_num <= 6):
        await msg.reply_text("player_num must be 1-6.")
        return

    if game_num == 0:
        field = "total"
    elif 1 <= game_num <= 6:
        field = f"game{game_num}"
        if not (0 <= value <= 10):
            await msg.reply_text("Game values must be between 0 and 10.")
            return
    else:
        await msg.reply_text("game_num must be 1-6, or 0 for total.")
        return

    ws_s = _week_start(datetime.now()).date().isoformat()
    con = _db()
    try:
        pending_row = _latest_pending_upload(con, ws_s, c.id, u.id)
        if not pending_row:
            logger.info(
                "fixscore_no_pending week_start=%s chat_id=%s user=%s",
                ws_s,
                c.id,
                _user_label(update),
            )
            await msg.reply_text("No pending upload found to fix (for this week).")
            return

        upload_id, image_path, csv_path = pending_row
        if not csv_path:
            await msg.reply_text("Pending upload has no CSV on record.")
            return

        csv_p = Path(csv_path)
        rows = _load_csv_rows(csv_p)

        matches = [r for r in rows if int(r.get("player_num", -1)) == player_num]
        if not matches:
            logger.info(
                "fixscore_player_not_found upload_id=%s P%s field=%s value=%s",
                upload_id,
                player_num,
                field,
                value,
            )
            await msg.reply_text(f"Couldn't find player number {player_num} in the pending CSV.")
            return

        # Update all matches (should be 1)
        for r in matches:
            old_val = r.get(field)
            r[field] = value
            logger.info(
                "fixscore_ok upload_id=%s week_start=%s chat_id=%s user=%s P%s %s: %s -> %s",
                upload_id,
                ws_s,
                c.id,
                _user_label(update),
                r.get("player_num"),
                field,
                old_val,
                value,
            )

        warns = _warnings_for_rows(rows)
        _write_csv_rows(csv_p, rows)

        con.execute(
            "UPDATE uploads SET warnings = ? WHERE id = ?",
            (warns or None, upload_id),
        )
        con.commit()

    finally:
        con.close()

    caption = (
        f"Updated CSV (PENDING upload #{upload_id}).\n"
        "Run /confirm when it looks good."
    )
    if warns:
        caption += f"\n\nWarnings:\n{warns}"

    await msg.reply_document(document=csv_p.open("rb"), filename=csv_p.name, caption=caption)


async def fixname(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg:
        return

    u = update.effective_user
    c = update.effective_chat
    if not u or not c:
        await msg.reply_text("Can’t determine user/chat.")
        return

    parts = shlex.split(msg.text or "")
    if len(parts) != 3:
        await msg.reply_text(
            "Usage: /fixname <player_num> <new_name>\n"
            "Example: /fixname 6 Anthony"
        )
        return

    _, player_num_s, new_name = parts

    try:
        player_num = int(player_num_s)
    except ValueError:
        await msg.reply_text("player_num must be an integer (1-6).")
        return

    if not (1 <= player_num <= 6):
        await msg.reply_text("player_num must be 1-6.")
        return

    ws_s = _week_start(datetime.now()).date().isoformat()
    con = _db()
    try:
        pending_row = _latest_pending_upload(con, ws_s, c.id, u.id)
        if not pending_row:
            logger.info(
                "fixname_no_pending week_start=%s chat_id=%s user=%s",
                ws_s,
                c.id,
                _user_label(update),
            )
            await msg.reply_text("No pending upload found to fix (for this week).")
            return

        upload_id, image_path, csv_path = pending_row
        if not csv_path:
            await msg.reply_text("Pending upload has no CSV on record.")
            return

        csv_p = Path(csv_path)
        rows = _load_csv_rows(csv_p)

        matches = [r for r in rows if int(r.get("player_num", -1)) == player_num]
        if not matches:
            logger.info(
                "fixname_player_not_found upload_id=%s P%s new_name=%s",
                upload_id,
                player_num,
                new_name,
            )
            await msg.reply_text(f"Couldn't find player number {player_num} in the pending CSV.")
            return

        for r in matches:
            old_name = r.get("player")
            r["player"] = new_name
            logger.info(
                "fixname_ok upload_id=%s week_start=%s chat_id=%s user=%s P%s: %s -> %s",
                upload_id,
                ws_s,
                c.id,
                _user_label(update),
                r.get("player_num"),
                old_name,
                new_name,
            )

        warns = _warnings_for_rows(rows)
        _write_csv_rows(csv_p, rows)

        con.execute(
            "UPDATE uploads SET warnings = ? WHERE id = ?",
            (warns or None, upload_id),
        )
        con.commit()

    finally:
        con.close()

    caption = (
        f"Updated CSV (PENDING upload #{upload_id}).\n"
        "Run /confirm when it looks good."
    )
    if warns:
        caption += f"\n\nWarnings:\n{warns}"

    await msg.reply_document(document=csv_p.open("rb"), filename=csv_p.name, caption=caption)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.photo:
        return

    await msg.chat.send_action(ChatAction.TYPING)

    # Get best resolution photo
    photo = msg.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    image_path_tmp = UPLOADS_DIR / f"scoresheet-{ts}.jpg"
    await file.download_to_drive(custom_path=str(image_path_tmp))

    # First extract team numbers so we can rename the file.
    team_home = None
    team_vis = None
    try:
        cmd_teams = [
            "python3",
            str(HERE / "extract_nil_sample_to_csv.py"),
            "--teams-only",
            "--image",
            str(image_path_tmp),
        ]
        proc_teams = subprocess.run(cmd_teams, cwd=str(HERE), capture_output=True, text=True)
        if proc_teams.returncode == 0:
            import json as _json

            teams = _json.loads((proc_teams.stdout or "").strip() or "{}")
            team_home = int(teams.get("home_team")) if "home_team" in teams else None
            team_vis = int(teams.get("visiting_team")) if "visiting_team" in teams else None
    except Exception:
        # If team extraction fails, continue with the tmp filename.
        team_home = None
        team_vis = None

    if team_home is not None and team_vis is not None:
        image_path = UPLOADS_DIR / f"scoresheet-{ts}-Team{team_home}vTeam{team_vis}.jpg"
        try:
            image_path_tmp.rename(image_path)
        except Exception:
            image_path = image_path_tmp
    else:
        image_path = image_path_tmp

    logger.info(
        "photo_received chat_id=%s user=%s file=%s teams=%s-%s",
        getattr(update.effective_chat, "id", None),
        _user_label(update),
        image_path.name,
        team_home,
        team_vis,
    )

    # Run extractor script (full)
    cmd = [
        "python3",
        str(HERE / "extract_nil_sample_to_csv.py"),
        "--image",
        str(image_path),
    ]

    proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    warn_text = ""
    if "WARNINGS:" in stdout:
        warn_text = stdout.split("WARNINGS:", 1)[1].strip()

    if proc.returncode != 0:
        # Never dump stack traces to users. Keep it actionable.
        friendly = (
            "Sorry — I couldn’t extract reliable scores from that photo.\n\n"
            "Please try again with:\n"
            "- less glare/shadow\n"
            "- photo straight-on\n"
            "- fill frame with the sheet\n\n"
            "If this keeps happening, send the same photo again and I’ll adjust the parser."
        )
        await msg.reply_text(friendly)
        return

    out_csv = HERE / "out" / f"{image_path.stem}.csv"
    if not out_csv.exists():
        await msg.reply_text("Extraction ran, but I couldn’t find the CSV output file.")
        return

    # Basic plausibility checks so random photos don't get a PENDING upload.
    try:
        rows = _load_csv_rows(out_csv)
    except Exception:
        await msg.reply_text(
            "Sorry — I got a CSV back but couldn’t read it. Please re-upload the scoresheet photo."
        )
        return

    teams_count = _teams_count()
    ok, problems = _plausibility_check(rows, teams_count, team_home, team_vis)
    if not ok:
        logger.info(
            "photo_rejected chat_id=%s user=%s file=%s problems=%s",
            getattr(update.effective_chat, "id", None),
            _user_label(update),
            image_path.name,
            "; ".join(problems),
        )
        details = "\n".join([f"- {p}" for p in problems])
        await msg.reply_text(
            "That doesn’t look like a NIL scoresheet upload, so I’m not going to accept it.\n\n"
            "If this *was* a scoresheet, try again with a clearer straight-on photo (less glare).\n\n"
            f"Checks that failed:\n{details}"
        )
        return

    # Record upload (PENDING until /confirm) — only after successful extraction + plausibility.
    upload_id = None
    con = _db()
    try:
        ws = _week_start(datetime.now()).date().isoformat()
        u = update.effective_user
        c = update.effective_chat
        cur = con.execute(
            """
            INSERT INTO uploads (
              created_at, week_start, chat_id, chat_title,
              user_id, username, first_name, last_name,
              image_path, csv_path, warnings, confirmed_at,
              home_team, visiting_team
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                ws,
                c.id if c else None,
                getattr(c, "title", None) if c else None,
                u.id if u else None,
                (u.username or None) if u else None,
                (u.first_name or None) if u else None,
                (u.last_name or None) if u else None,
                str(image_path),
                str(out_csv),
                warn_text or None,
                None,
                team_home,
                team_vis,
            ),
        )
        upload_id = cur.lastrowid
        con.commit()
    finally:
        con.close()

    logger.info(
        "upload_recorded upload_id=%s week_start=%s chat_id=%s user=%s home_team=%s visiting_team=%s csv=%s warnings=%s",
        upload_id,
        ws,
        getattr(update.effective_chat, "id", None),
        _user_label(update),
        team_home,
        team_vis,
        out_csv.name,
        "yes" if warn_text else "no",
    )

    teams_line = ""
    if team_home is not None and team_vis is not None:
        teams_line = f"Teams: Home Team {team_home} vs Visiting Team {team_vis}\n"

    caption = (
        f"Processed photo: {image_path.name}\n"
        f"{teams_line}"
        f"Parsed CSV (PENDING upload #{upload_id}).\n"
        "Please /confirm if this is good; otherwise fix issues and re-upload."
    )
    if warn_text:
        caption += f"\n\nWarnings:\n{warn_text}"

    await msg.reply_document(document=out_csv.open("rb"), filename=out_csv.name, caption=caption)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN env var")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler(["start"], start))
    app.add_handler(CommandHandler(["help"], help_cmd))
    app.add_handler(CommandHandler(["confirm"], confirm))
    app.add_handler(CommandHandler(["fixscore"], fixscore))
    app.add_handler(CommandHandler(["fixname"], fixname))
    app.add_handler(CommandHandler(["status"], status))
    app.add_handler(CommandHandler(["pending"], pending))
    app.add_handler(CommandHandler(["recent"], recent))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
