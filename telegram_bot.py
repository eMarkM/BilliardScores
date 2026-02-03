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

import os
import sqlite3
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

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
    "3) If not, fix issues and re-upload a clearer photo\n\n"
    "Commands:\n"
    "- /confirm — confirm your latest pending upload for this week\n"
    "- /status — show confirmed uploads since Monday\n"
    "- /pending — show pending uploads since Monday\n"
    "- /recent — show recent uploads\n\n"
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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT)


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

    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler(["confirm"], confirm))
    app.add_handler(CommandHandler(["status"], status))
    app.add_handler(CommandHandler(["pending"], pending))
    app.add_handler(CommandHandler(["recent"], recent))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
