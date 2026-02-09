from telegram_bot import _friendly_extract_failure


def test_friendly_extract_failure_mentions_player_when_present():
    msg = _friendly_extract_failure("ERROR: Extraction failed: Could not process home player 1")
    assert "home team player 1" in msg.lower()
    assert "fill the frame with the box scores" in msg.lower()


def test_friendly_extract_failure_generic_when_unknown():
    msg = _friendly_extract_failure("ERROR: Extraction failed: some other error")
    assert "couldn’t extract reliable scores" in msg.lower()
    assert "fill the frame with the box scores" in msg.lower()
