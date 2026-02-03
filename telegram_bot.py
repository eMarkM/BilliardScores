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


HELP_TEXT = (
    "Send me a clear photo of the NIL scoresheet and I’ll reply with a CSV.\n\n"
    "Workflow:\n"
    "1) Send photo → I reply with a CSV and mark it PENDING\n"
    "2) If it looks good, reply /confirm to lock it in\n"
    "3) If not, use /fixscore or /fixname (or re-upload a clearer photo)\n\n"
    "Commands:\n"
    "- /confirm — confirm your latest pending upload for this week\n"
    "- /status — show confirmed uploads since Monday\n"
    "- /fixscore — correct a score in your pending CSV\n"
    "- /fixname — correct a player name in your pending CSV\n"
    "- /pending — show pending uploads since Monday\n"
    "- /recent — show recent uploads\n"
    "- /help — show this help\n\n"
    "Fix score format:\n"
    "  /fixscore <home|visiting> <player> <game1..game6|total> <value>\n"
    "Examples:\n"
    "  /fixscore visiting Ed game2 4\n"
    "  /fixscore home Sue game4 5\n\n"
    "Fix name format:\n"
    "  /fixname <home|visiting> <old_name> <new_name>\n"
    "Example:\n"
    "  /fixname visiting \"Jam\" Joe\n\n"
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
          confirmed_at TEXT
        )
        """
    )

    # Lightweight migrations for older DBs
    try:
        con.execute("ALTER TABLE uploads ADD COLUMN confirmed_at TEXT")
        con.commit()
    except sqlite3.OperationalError:
        pass

    return con


def _week_start(dt: datetime) -> datetime:
    # Monday 00:00 local
    d0 = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return d0 - timedelta(days=d0.weekday())


def _expected_captains() -> list[str]:
    # comma-separated telegram usernames, no @, e.g. "edmay,randy,dave"
    raw = (os.getenv("CAPTAINS") or "").strip()
    if not raw:
        return []
    return [x.strip().lstrip("@").lower() for x in raw.split(",") if x.strip()]


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
            SELECT created_at, username, first_name, last_name, id
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
    for created_at, username, first_name, last_name, upload_id in rows:
        who = f"@{username}" if username else " ".join([x for x in [first_name, last_name] if x])
        lines.append(f"- #{upload_id} {created_at}: {who}")

    await update.message.reply_text("\n".join(lines))


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    now = datetime.now()
    ws_s = _week_start(now).date().isoformat()

    con = _db()
    try:
        uploads = con.execute(
            """
            SELECT DISTINCT COALESCE(LOWER(username), '') AS u
            FROM uploads
            WHERE week_start = ? AND confirmed_at IS NOT NULL
            """,
            (ws_s,),
        ).fetchall()
    finally:
        con.close()

    uploaded_usernames = sorted([u for (u,) in uploads if u])

    lines = [f"Status (CONFIRMED) for week starting {ws_s}:"]
    if uploaded_usernames:
        lines.append("Confirmed uploads:")
        lines.extend([f"- @{u}" for u in uploaded_usernames])
    else:
        lines.append("No confirmed uploads yet this week.")

    expected = _expected_captains()
    if expected:
        missing = [u for u in expected if u not in uploaded_usernames]
        lines.append("")
        lines.append("Expected captains:")
        lines.extend([f"- @{u}" for u in expected])
        lines.append("")
        lines.append("Missing confirmed uploads:")
        if missing:
            lines.extend([f"- @{u}" for u in missing])
        else:
            lines.append("- (none)")
    else:
        lines.append("")
        lines.append("Tip: set CAPTAINS in .env to track missing uploads.")

    await update.message.reply_text("\n".join(lines))


def _load_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = [dict(row) for row in r]
    # normalize ints
    for row in rows:
        for k in list(row.keys()):
            if k.startswith("game") or k == "total":
                row[k] = int(row[k])
    return rows


def _write_csv_rows(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ["side", "player", "game1", "game2", "game3", "game4", "game5", "game6", "total"]
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
            warns.append(f"Total mismatch for {row['side']}:{row['player']}: games sum={s} total={row['total']}")
    return "\n".join(warns)


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
    if len(parts) != 5:
        await msg.reply_text(
            "Usage: /fixscore <home|visiting> <player> <game1..game6|total> <value>\n"
            "Example: /fixscore visiting Ed game2 4"
        )
        return

    _, side, player, field, value_s = parts
    side = side.lower()
    field = field.lower()

    if side not in ("home", "visiting"):
        await msg.reply_text("Side must be 'home' or 'visiting'.")
        return

    if field not in {"game1", "game2", "game3", "game4", "game5", "game6", "total"}:
        await msg.reply_text("Field must be game1..game6 or total.")
        return

    try:
        value = int(value_s)
    except ValueError:
        await msg.reply_text("Value must be an integer.")
        return

    if field.startswith("game") and not (0 <= value <= 10):
        await msg.reply_text("Game values must be between 0 and 10.")
        return

    ws_s = _week_start(datetime.now()).date().isoformat()
    con = _db()
    try:
        pending_row = _latest_pending_upload(con, ws_s, c.id, u.id)
        if not pending_row:
            await msg.reply_text("No pending upload found to fix (for this week).")
            return

        upload_id, image_path, csv_path = pending_row
        if not csv_path:
            await msg.reply_text("Pending upload has no CSV on record.")
            return

        csv_p = Path(csv_path)
        rows = _load_csv_rows(csv_p)

        matches = [r for r in rows if r["side"].lower() == side and r["player"] == player]
        if not matches:
            await msg.reply_text(f"Couldn't find player '{player}' on side '{side}' in the pending CSV.")
            return

        # Update all matches (should be 1)
        for r in matches:
            r[field] = value

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
    if len(parts) != 4:
        await msg.reply_text(
            "Usage: /fixname <home|visiting> <old_name> <new_name>\n"
            "Example: /fixname visiting \"3rd A H\" Anthony"
        )
        return

    _, side, old_name, new_name = parts
    side = side.lower()

    if side not in ("home", "visiting"):
        await msg.reply_text("Side must be 'home' or 'visiting'.")
        return

    ws_s = _week_start(datetime.now()).date().isoformat()
    con = _db()
    try:
        pending_row = _latest_pending_upload(con, ws_s, c.id, u.id)
        if not pending_row:
            await msg.reply_text("No pending upload found to fix (for this week).")
            return

        upload_id, image_path, csv_path = pending_row
        if not csv_path:
            await msg.reply_text("Pending upload has no CSV on record.")
            return

        csv_p = Path(csv_path)
        rows = _load_csv_rows(csv_p)

        matches = [r for r in rows if r["side"].lower() == side and r["player"] == old_name]
        if not matches:
            await msg.reply_text(f"Couldn't find player '{old_name}' on side '{side}' in the pending CSV.")
            return

        for r in matches:
            r["player"] = new_name

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
    image_path = UPLOADS_DIR / f"scoresheet-{ts}.jpg"
    await file.download_to_drive(custom_path=str(image_path))

    # Run extractor script
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

    # Record upload (PENDING until /confirm) — only after successful extraction
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
              image_path, csv_path, warnings, confirmed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                str(out_csv) if out_csv.exists() else None,
                warn_text or None,
                None,
            ),
        )
        upload_id = cur.lastrowid
        con.commit()
    finally:
        con.close()

    caption = (
        f"Processed photo: {image_path.name}\n"
        f"Parsed CSV (PENDING upload #{upload_id}).\n"
        "Please /confirm if this is good; otherwise fix issues and re-upload."
    )
    if warn_text:
        caption += f"\n\nWarnings:\n{warn_text}"

    if out_csv.exists():
        await msg.reply_document(document=out_csv.open("rb"), filename=out_csv.name, caption=caption)
    else:
        await msg.reply_text("Extraction ran, but I couldn’t find the CSV output file.")


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
