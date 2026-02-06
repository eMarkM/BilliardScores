from __future__ import annotations

from PIL import Image, ImageDraw

from extract_nil_sample_to_csv import detect_row_bands_by_grid


def test_detect_row_bands_by_grid_picks_three_tall_bands():
    # Synthetic "boxscore" with horizontal grid lines.
    w, h = 1280, 960
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)

    # Draw header separator near 0.16h
    header_y = int(h * 0.16)
    d.line((0, header_y, w, header_y), fill=0, width=2)

    # For each of 3 players: score band (~90px), mark (~30px), opp (~30px)
    y = header_y
    for _ in range(3):
        y += 90
        d.line((0, y, w, y), fill=0, width=2)
        y += 30
        d.line((0, y, w, y), fill=0, width=2)
        y += 30
        d.line((0, y, w, y), fill=0, width=2)

    bands = detect_row_bands_by_grid(img)
    assert bands is not None
    assert "home" in bands and "visiting" in bands
    assert len(bands["home"]["rows"]) == 3

    # Ensure we didn't pick the short bands (mark/opponents)
    rows = bands["home"]["rows"]
    heights = [r["y2"] - r["y1"] for r in rows]
    assert all(hh > 0.05 for hh in heights)
