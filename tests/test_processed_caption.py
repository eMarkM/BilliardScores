from bot_core import build_processed_caption


def test_build_processed_caption_includes_team_and_message_id():
    caption = build_processed_caption(
        image_filename="IMG_1234.jpg",
        message_id="c25732bf-7223-4b8a-8954-75c50c5f5394",
        home_team=7,
        warn_text=None,
    )

    assert "Team 7 has been successfully processed." in caption
    assert "send /confirm" in caption
    assert "Use /fixname and /fixscore" in caption
    assert "Processed photo: IMG_1234.jpg" in caption
    assert "[message_id: c25732bf-7223-4b8a-8954-75c50c5f5394]" in caption


def test_build_processed_caption_omits_message_id_when_missing():
    caption = build_processed_caption(
        image_filename="IMG_1234.jpg",
        message_id=None,
        home_team=7,
        warn_text=None,
    )
    assert "[message_id:" not in caption


def test_build_processed_caption_appends_warnings():
    caption = build_processed_caption(
        image_filename="IMG_1234.jpg",
        message_id="123",
        home_team=7,
        warn_text="Total mismatch for P1",
    )

    assert "Warnings:" in caption
    assert "Total mismatch for P1" in caption


def test_build_processed_caption_handles_missing_team():
    caption = build_processed_caption(
        image_filename="IMG_1234.jpg",
        message_id="123",
        home_team=None,
        warn_text=None,
    )

    assert caption.startswith("Scoresheet has been successfully processed.")
