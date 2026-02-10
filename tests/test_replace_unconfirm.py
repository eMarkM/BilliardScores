from __future__ import annotations

from datetime import datetime

import telegram_bot


def test_unconfirm_latest_confirmed_upload_sets_back_to_pending(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(telegram_bot, "DB_PATH", db_path)

    ws_s = "2026-02-09"
    chat_id = 123
    user_id = 456

    con = telegram_bot._db()
    try:
        # Insert a confirmed upload
        con.execute(
            """
            INSERT INTO uploads (
              created_at, week_start, chat_id, user_id, image_path, csv_path, confirmed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                ws_s,
                chat_id,
                user_id,
                "/tmp/scoresheet.jpg",
                "/tmp/scoresheet.csv",
                datetime.now().isoformat(),
            ),
        )
        con.commit()

        confirmed = telegram_bot._latest_confirmed_upload(con, ws_s, chat_id, user_id)
        assert confirmed is not None
        upload_id, *_rest = confirmed

        # Act
        unconfirmed_id = telegram_bot._unconfirm_latest_confirmed_upload(con, ws_s, chat_id, user_id)

        assert unconfirmed_id == upload_id

        # Now there should be no confirmed upload, and the row should be pending.
        assert telegram_bot._latest_confirmed_upload(con, ws_s, chat_id, user_id) is None

        pending = con.execute(
            "SELECT id, confirmed_at FROM uploads WHERE id = ?",
            (upload_id,),
        ).fetchone()
        assert pending == (upload_id, None)
    finally:
        con.close()
