from __future__ import annotations

from datetime import datetime

import telegram_bot


def test_unconfirm_all_confirmed_uploads_sets_back_to_pending(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(telegram_bot, "DB_PATH", db_path)

    ws_s = "2026-02-09"
    chat_id = 123
    user_id = 456

    con = telegram_bot._db()
    try:
        # Insert two confirmed uploads (older + newer)
        con.execute(
            """
            INSERT INTO uploads (
              created_at, week_start, chat_id, user_id, image_path, csv_path, confirmed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-02-09T10:00:00",
                ws_s,
                chat_id,
                user_id,
                "/tmp/old.jpg",
                "/tmp/old.csv",
                "2026-02-09T10:01:00",
            ),
        )
        con.execute(
            """
            INSERT INTO uploads (
              created_at, week_start, chat_id, user_id, image_path, csv_path, confirmed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-02-09T11:00:00",
                ws_s,
                chat_id,
                user_id,
                "/tmp/new.jpg",
                "/tmp/new.csv",
                "2026-02-09T11:01:00",
            ),
        )
        con.commit()

        latest = telegram_bot._latest_confirmed_upload(con, ws_s, chat_id, user_id)
        assert latest is not None
        latest_id = int(latest[0])

        # Act
        count, most_recent_id = telegram_bot._unconfirm_all_confirmed_uploads(con, ws_s, chat_id, user_id)

        assert count == 2
        assert most_recent_id == latest_id

        # Now there should be no confirmed uploads.
        assert telegram_bot._latest_confirmed_upload(con, ws_s, chat_id, user_id) is None

        rows = con.execute(
            "SELECT confirmed_at FROM uploads WHERE week_start=? AND chat_id=? AND user_id=? ORDER BY id",
            (ws_s, chat_id, user_id),
        ).fetchall()
        assert rows == [(None,), (None,)]
    finally:
        con.close()
