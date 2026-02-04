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
  python3 src/telegram_bot.py
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
import shlex
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from bot_core import (
    week_start,
    load_csv_rows,
    write_csv_rows,
    warnings_for_rows,
    plausibility_check,
    apply_fixname,
    apply_fixscore,
)

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

UPLOADS_DIR = PROJECT_ROOT / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = PROJECT_ROOT / "bot.db"
LOG_PATH = PROJECT_ROOT / "bot.log"

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


# If a user already has a confirmed upload for the week, require /replace before accepting a new photo.
REPLACE_ARMED: Dict[Tuple[int, int, str], bool] = {}


HELP_TEXT = (
    "Send me a clear photo of the NIL scoresheet and I’ll reply with a CSV.\n\n"
    "Workflow:\n"
    "1) Send photo → I reply with a CSV and mark it PENDING\n"
    "2) If it looks good, reply /confirm to lock it in\n"
    "3) If not, use /fixscore or /fixname (or re-upload a clearer photo)\n\n"
    "Commands:\n"
    "- /confirm — confirm your latest pending upload for this week\n"
    "- /csv — get your confirmed CSV for this week\n"
    "- /replace — allow replacing your confirmed upload (then send new photo)\n"
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
    return week_start(dt)


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


# CSV + validation logic lives in bot_core.py (unit-tested)


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


def _latest_confirmed_upload(con: sqlite3.Connection, ws_s: str, chat_id: int, user_id: int):
    return con.execute(
        """
        SELECT id, created_at, image_path, csv_path
        FROM uploads
        WHERE week_start = ? AND chat_id = ? AND user_id = ? AND confirmed_at IS NOT NULL
        ORDER BY id DESC
        LIMIT 1
        """,
        (ws_s, chat_id, user_id),
    ).fetchone()


async def replace_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg:
        return

    u = update.effective_user
    c = update.effective_chat
    if not u or not c:
        await msg.reply_text("Can’t determine user/chat.")
        return

    ws_s = _week_start(datetime.now()).date().isoformat()
    # Arm replace for this week in this chat/user
    REPLACE_ARMED[(c.id, u.id, ws_s)] = True
    logger.info("replace_armed week_start=%s chat_id=%s user=%s", ws_s, c.id, _user_label(update))
    await msg.reply_text(
        "OK — send the updated scoresheet photo now and I’ll treat it as a replacement for this week’s confirmed upload."
    )


async def csv_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        row = _latest_confirmed_upload(con, ws_s, c.id, u.id)
    finally:
        con.close()

    if not row:
        logger.info("csv_no_confirmed week_start=%s chat_id=%s user=%s", ws_s, c.id, _user_label(update))
        await msg.reply_text("No confirmed upload found for this week yet.")
        return

    upload_id, created_at, image_path, csv_path = row
    if not csv_path or not Path(csv_path).exists():
        await msg.reply_text("I found your confirmed upload, but the CSV file is missing.")
        return

    logger.info(
        "csv_send upload_id=%s week_start=%s chat_id=%s user=%s",
        upload_id,
        ws_s,
        c.id,
        _user_label(update),
    )

    caption = f"Confirmed CSV for week starting {ws_s} (upload #{upload_id})."
    await msg.reply_document(document=Path(csv_path).open("rb"), filename=Path(csv_path).name, caption=caption)


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
        rows = load_csv_rows(csv_p)

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

        ok2, err2 = apply_fixscore(rows, player_num, game_num, value)
        if not ok2:
            await msg.reply_text(err2)
            return

        logger.info(
            "fixscore_ok upload_id=%s week_start=%s chat_id=%s user=%s P%s game=%s value=%s",
            upload_id,
            ws_s,
            c.id,
            _user_label(update),
            player_num,
            game_num,
            value,
        )

        warns = warnings_for_rows(rows)
        write_csv_rows(csv_p, rows)

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
        rows = load_csv_rows(csv_p)

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

        ok2, err2 = apply_fixname(rows, player_num, new_name)
        if not ok2:
            await msg.reply_text(err2)
            return

        logger.info(
            "fixname_ok upload_id=%s week_start=%s chat_id=%s user=%s P%s -> %s",
            upload_id,
            ws_s,
            c.id,
            _user_label(update),
            player_num,
            new_name,
        )

        warns = warnings_for_rows(rows)
        write_csv_rows(csv_p, rows)

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

    # If user already has a confirmed upload this week, require /replace before processing.
    ws_s = _week_start(datetime.now()).date().isoformat()
    con_warn = _db()
    try:
        confirmed = _latest_confirmed_upload(
            con_warn,
            ws_s,
            update.effective_chat.id if update.effective_chat else 0,
            update.effective_user.id if update.effective_user else 0,
        )
    finally:
        con_warn.close()

    if confirmed:
        cid, created_at, img_path, csv_path = confirmed
        armed = REPLACE_ARMED.pop((update.effective_chat.id, update.effective_user.id, ws_s), False)
        if not armed:
            logger.info(
                "upload_blocked_needs_replace upload_id=%s week_start=%s chat_id=%s user=%s",
                cid,
                ws_s,
                update.effective_chat.id,
                _user_label(update),
            )
            await msg.reply_text(
                f"You already have a CONFIRMED upload for this week (upload #{cid}).\n\n"
                "If you want to replace it, run /replace and then send the new photo again."
            )
            return
        else:
            logger.info(
                "upload_replace_allowed confirmed_upload_id=%s week_start=%s chat_id=%s user=%s",
                cid,
                ws_s,
                update.effective_chat.id,
                _user_label(update),
            )

    # Get best resolution photo
    photo = msg.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    image_path_tmp = UPLOADS_DIR / f"scoresheet-{ts}.jpg"
    await file.download_to_drive(custom_path=str(image_path_tmp))

    # First extract team numbers so we can rename the file.
    team_home = None
    team_vis = None
    cmd_teams = [
        "python3",
        str(HERE / "extract_nil_sample_to_csv.py"),
        "--teams-only",
        "--image",
        str(image_path_tmp),
    ]
    try:
        proc_teams = subprocess.run(cmd_teams, cwd=str(HERE), capture_output=True, text=True)
        if proc_teams.returncode != 0:
            logger.info(
                "team_extract_failed chat_id=%s user=%s file=%s rc=%s stdout=%s stderr=%s",
                getattr(update.effective_chat, "id", None),
                _user_label(update),
                image_path_tmp.name,
                proc_teams.returncode,
                (proc_teams.stdout or "").strip()[:500],
                (proc_teams.stderr or "").strip()[:500],
            )
        else:
            import json as _json

            teams = _json.loads((proc_teams.stdout or "").strip() or "{}")
            team_home = int(teams.get("home_team")) if "home_team" in teams else None
            team_vis = int(teams.get("visiting_team")) if "visiting_team" in teams else None
    except Exception:
        logger.exception(
            "team_extract_exception chat_id=%s user=%s file=%s",
            getattr(update.effective_chat, "id", None),
            _user_label(update),
            image_path_tmp.name,
        )
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
    debug_dir = PROJECT_ROOT / "out" / f"debug-{image_path.stem}"
    cmd = [
        "python3",
        str(HERE / "extract_nil_sample_to_csv.py"),
        "--image",
        str(image_path),
        "--debug-dir",
        str(debug_dir),
    ]

    proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    warn_text = ""
    if "WARNINGS:" in stdout:
        warn_text = stdout.split("WARNINGS:", 1)[1].strip()

    if proc.returncode != 0:
        logger.info(
            "extract_failed chat_id=%s user=%s file=%s rc=%s debug_dir=%s stdout=%s stderr=%s",
            getattr(update.effective_chat, "id", None),
            _user_label(update),
            image_path.name,
            proc.returncode,
            str(debug_dir),
            stdout[:500],
            stderr[:500],
        )
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

    out_csv = PROJECT_ROOT / "out" / f"{image_path.stem}.csv"
    if not out_csv.exists():
        await msg.reply_text("Extraction ran, but I couldn’t find the CSV output file.")
        return

    # Basic plausibility checks so random photos don't get a PENDING upload.
    try:
        rows = load_csv_rows(out_csv)
    except Exception:
        await msg.reply_text(
            "Sorry — I got a CSV back but couldn’t read it. Please re-upload the scoresheet photo."
        )
        return

    teams_count = _teams_count()
    ok, problems = plausibility_check(rows, teams_count, team_home, team_vis)
    if not ok:
        logger.info(
            "photo_rejected chat_id=%s user=%s file=%s teams=%s-%s problems=%s",
            getattr(update.effective_chat, "id", None),
            _user_label(update),
            image_path.name,
            team_home,
            team_vis,
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
        ws = week_start(datetime.now()).date().isoformat()
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

    logger.info(
        "service_started pid=%s cwd=%s db=%s uploads=%s out=%s",
        os.getpid(),
        os.getcwd(),
        DB_PATH,
        UPLOADS_DIR,
        PROJECT_ROOT / "out",
    )

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler(["start"], start))
    app.add_handler(CommandHandler(["help"], help_cmd))
    app.add_handler(CommandHandler(["confirm"], confirm))
    app.add_handler(CommandHandler(["csv"], csv_cmd))
    app.add_handler(CommandHandler(["replace"], replace_cmd))
    app.add_handler(CommandHandler(["fixscore"], fixscore))
    app.add_handler(CommandHandler(["fixname"], fixname))
    app.add_handler(CommandHandler(["status"], status))
    app.add_handler(CommandHandler(["pending"], pending))
    app.add_handler(CommandHandler(["recent"], recent))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
