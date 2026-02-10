from __future__ import annotations

from PIL import Image, ImageDraw

from extract_nil_sample_to_csv import _has_two_horizontal_border_lines


def test_has_two_horizontal_border_lines_detects_two_lines():
    img = Image.new("L", (400, 120), color=245)
    draw = ImageDraw.Draw(img)
    # Two thick dark lines
    draw.rectangle((0, 20, 399, 24), fill=10)
    draw.rectangle((0, 95, 399, 99), fill=10)

    assert _has_two_horizontal_border_lines(img) is True


def test_has_two_horizontal_border_lines_rejects_single_line():
    img = Image.new("L", (400, 120), color=245)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 20, 399, 24), fill=10)

    assert _has_two_horizontal_border_lines(img) is False
