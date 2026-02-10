#!/usr/bin/env python3
"""Extract NIL scoresheet player scores to CSV.

v1: Vision-model extraction from an image (handwriting-friendly) + basic validation.

Update: uses row-level zoom/cropping to dramatically improve accuracy.

Requirements:
  pip install -r requirements.txt

Auth:
  export OPENAI_API_KEY=...   # required to run vision extraction

Usage:
  python3 src/extract_nil_sample_to_csv.py --image ../NilSample.jpeg

Outputs:
  out/<image>.extracted.json  (raw model output: 6 row objects after normalization)
  out/<image>.csv             (normalized CSV)

Notes:
  - This first pass extracts only: player, rating, games 1-6, total, side.
  - Marks (BR/TR/WZ/WF), opponents, handicap grids, etc. are deferred.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import base64
import csv
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger("nilpoolbot.extractor")
if not logger.handlers:
    _h = logging.StreamHandler(stream=sys.stderr)
    _h.setFormatter(logging.Formatter("EXTRACTOR %(levelname)s %(message)s"))
    logger.addHandler(_h)
logger.setLevel(getattr(logging, (os.getenv("BILLIARDSCORES_LOG_LEVEL") or "INFO").upper(), logging.INFO))


def _has_two_horizontal_border_lines(crop: Image.Image) -> bool:
    """Heuristic: does the crop include two strong horizontal table border lines?

    This is meant to prevent "half row" crops where we cut off the top or bottom
    of the score band.

    We look for two dark horizontal peaks: one in the top half, one in the bottom
    half, separated by a reasonable distance.
    """

    w, h = crop.size
    if w < 200 or h < 40:
        return True  # Too small to judge; don't block.

    # Work in grayscale and lightly blur to reduce noise.
    g = crop.convert("L").filter(ImageFilter.GaussianBlur(radius=0.8))
    px = list(g.getdata())

    # Mean brightness per row (0=black, 255=white)
    row_means: List[float] = []
    for y in range(h):
        start = y * w
        row = px[start : start + w]
        row_means.append(sum(row) / float(w))

    # Smooth a bit (moving average)
    smoothed: List[float] = []
    for y in range(h):
        y0 = max(0, y - 2)
        y1 = min(h, y + 3)
        smoothed.append(sum(row_means[y0:y1]) / float(y1 - y0))

    # Establish a relative darkness threshold.
    sorted_vals = sorted(smoothed)
    p10 = sorted_vals[int(0.10 * (h - 1))]
    median = sorted_vals[int(0.50 * (h - 1))]

    # Border lines should be noticeably darker than the median row.
    thresh = min(p10 + 10.0, median - 18.0)

    candidates = [y for y, v in enumerate(smoothed) if v <= thresh]
    if not candidates:
        return False

    top_region = [y for y in candidates if y <= int(h * 0.55)]
    bot_region = [y for y in candidates if y >= int(h * 0.45)]
    if not top_region or not bot_region:
        return False

    top_y = min(top_region, key=lambda y: smoothed[y])
    bot_y = min(bot_region, key=lambda y: smoothed[y])

    # Must be separated enough to plausibly be two borders.
    return (bot_y - top_y) >= int(h * 0.35)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

OUT_DIR = PROJECT_ROOT / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimal, demo-friendly schema
FIELDS = [
    "player_num",
    "side",
    "player",
    "game1",
    "game2",
    "game3",
    "game4",
    "game5",
    "game6",
    "total",
]


ROW_PROMPT = """You are extracting ONE player SCORE row from a pool league scoresheet photo.

This is a sports scoresheet (not an ID document). Do NOT identify real people beyond copying the handwritten name as it appears.

Key layout cue:
- In the PLAYER NAME box, there is a small printed row index number in the lower-left corner (1,2,3,4,5,6), right above the printed "mark BR TR" line.
- The handwritten player name starts immediately to the RIGHT of that small number.
- Ignore any printed helper text like "mark", "BR", "TR", "opponents", and ignore any matchup strings like "1v4".

IMPORTANT: This crop may include parts of adjacent rows (like the "mark" line) and may include the TOTAL box at the far right.
- game1..game6 are the SIX game columns labeled 1,2,3,4,5,6
- total is the separate TOTAL column at the far right
- Do NOT copy the total into game6.

Output STRICT JSON with keys:
- has_opponents_word: boolean (optional; true if the printed word "opponents" appears anywhere in this crop)
- has_mark_word: boolean (optional; true if the printed word "mark" appears anywhere in this crop)
- player: string (copy the handwritten name as written)
- game1..game6: integers 0-7 or 10 ONLY (the six GAME columns, in order)
- total: integer (TOTAL column)

Rules:
- Output MUST be a single JSON object.
- Only include those keys.
- If you can see "opponents" or "mark" in the crop, set the corresponding boolean true.
- Valid game scores are ONLY 0,1,2,3,4,5,6,7,10. 8 and 9 are impossible.
- total should equal sum(game1..game6). If it doesn't, re-check the digits.
"""

# Row crop boxes captured from a representative phone photo after rotating it
# to an upright/landscape orientation (width > height). We scale these to
# whatever resolution the rotated image is.
BASE_W, BASE_H = 1280, 960

# Team numbers in header (tight-ish boxes around the handwritten digits)
HOME_TEAM_BOX_BASE: Tuple[int, int, int, int] = (340, 105, 370, 140)
VISITING_TEAM_BOX_BASE: Tuple[int, int, int, int] = (790, 105, 820, 140)

# Tuned after debug-crop inspection. These boxes are applied to a normalized
# (BASE_W x BASE_H) crop of the boxscore region.
#
# Goal: capture ONLY the main score row (rating/player + games 1-6 + total),
# not the "mark BR, TR..." or "opponents" rows.
HOME_ROW_BOXES_BASE: List[Tuple[int, int, int, int]] = [
    # Row 1 score band (David)
    (0, 215, 610, 285),
    # Row 2 score band (Anthony)
    (0, 515, 610, 595),
    # Row 3 score band (Ed) — push lower
    (0, 730, 610, 810),
]
VISITING_ROW_BOXES_BASE: List[Tuple[int, int, int, int]] = [
    (620, 215, 1280, 285),
    (620, 515, 1280, 595),
    (620, 730, 1280, 810),
]


def _b64_data_url_bytes(img_bytes: bytes, mime: str) -> str:
    data = base64.b64encode(img_bytes).decode("ascii")
    return f"data:{mime};base64,{data}"


def _img_bytes(img: Image.Image, *, max_w: int = 1280, fmt: str = "JPEG") -> tuple[bytes, str]:
    """Encode an image for sending to the vision model.

    Downscales large images to keep payload size reasonable.
    Returns (bytes, mime).
    """

    im = img
    if im.width > max_w:
        scale = max_w / im.width
        im = im.resize((max_w, int(im.height * scale)), resample=Image.BILINEAR)

    import io

    bio = io.BytesIO()
    if fmt.upper() == "JPEG":
        im.convert("RGB").save(bio, format="JPEG", quality=85, optimize=True)
        return bio.getvalue(), "image/jpeg"
    im.save(bio, format="PNG")
    return bio.getvalue(), "image/png"


def _img_crop_bytes(
    img: Image.Image,
    box: Tuple[int, int, int, int],
    *,
    upscale: int = 1,
) -> bytes:
    crop = img.crop(box)
    if upscale and upscale > 1:
        crop = crop.resize((crop.width * upscale, crop.height * upscale), resample=Image.NEAREST)

    import io

    bio = io.BytesIO()
    # Use PNG for crops (lossless, keeps sharp digits)
    crop.save(bio, format="PNG")
    return bio.getvalue()


def _scale_box(box: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    sx = w / BASE_W
    sy = h / BASE_H
    x1, y1, x2, y2 = box
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t.replace("json\n", "", 1).strip()
    return t


class VisionRefusal(RuntimeError):
    pass


TEAM_SCHEMA = {
    "name": "nil_scoresheet_teams",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "home_team": {"type": "integer"},
            "visiting_team": {"type": "integer"},
        },
        "required": ["home_team", "visiting_team"],
    },
}

BOX_SCHEMA = {
    "name": "nil_scoresheet_box",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x1": {"type": "number"},
            "y1": {"type": "number"},
            "x2": {"type": "number"},
            "y2": {"type": "number"},
            "rotation": {
                "type": "integer",
                "enum": [0, 180],
                "description": "0 if image is upright, 180 if upside-down",
            },
        },
        "required": ["x1", "y1", "x2", "y2", "rotation"],
    },
}

DATE_SCHEMA = {
    "name": "nil_scoresheet_date_box",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x1": {"type": "number"},
            "y1": {"type": "number"},
            "x2": {"type": "number"},
            "y2": {"type": "number"},
        },
        "required": ["x1", "y1", "x2", "y2"],
    },
}

ROW_SCHEMA = {
    "name": "nil_scoresheet_row",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "has_opponents_word": {"type": "boolean"},
            "has_mark_word": {"type": "boolean"},
            "player": {"type": "string"},
            "game1": {"type": "integer"},
            "game2": {"type": "integer"},
            "game3": {"type": "integer"},
            "game4": {"type": "integer"},
            "game5": {"type": "integer"},
            "game6": {"type": "integer"},
            "total": {"type": "integer"},
        },
        "required": [
            "player",
            "game1",
            "game2",
            "game3",
            "game4",
            "game5",
            "game6",
            "total",
        ],
    },
}

ROWBANDS_SCHEMA = {
    "name": "nil_scoresheet_rowbands",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "header_y2": {
                "type": "number",
                "description": "Normalized y (0..1) of the bottom edge of the printed header row (Rating/Player/1..6/Total)",
            },
            "home": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "x1": {"type": "number"},
                    "x2": {"type": "number"},
                    "rows": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "y1": {"type": "number"},
                                "y2": {"type": "number"},
                            },
                            "required": ["y1", "y2"],
                        },
                    },
                },
                "required": ["x1", "x2", "rows"],
            },
            "visiting": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "x1": {"type": "number"},
                    "x2": {"type": "number"},
                    "rows": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "y1": {"type": "number"},
                                "y2": {"type": "number"},
                            },
                            "required": ["y1", "y2"],
                        },
                    },
                },
                "required": ["x1", "x2", "rows"],
            },
        },
        "required": ["header_y2", "home", "visiting"],
    },
}


from vision_client import VisionClient, OpenAIVisionClient


def openai_vision_json(prompt: str, data_url: str, model: str, schema: dict) -> Any:
    """Backward-compatible helper (live OpenAI call)."""

    return OpenAIVisionClient().vision_json(prompt=prompt, data_url=data_url, model=model, schema=schema)


def vision_json(client: VisionClient, prompt: str, data_url: str, model: str, schema: dict) -> Any:
    """Pluggable helper used by the extractor pipeline."""

    obj = client.vision_json(prompt=prompt, data_url=data_url, model=model, schema=schema)
    return obj


def _load_upright(image_path: Path) -> Image.Image:
    """Load image and normalize orientation.

    Captains will send portrait photos; the scoresheet itself is landscape.
    We rotate portrait images 90° so width > height.
    """

    img = ImageOps.exif_transpose(Image.open(image_path))
    w, h = img.size
    if h > w:
        img = img.rotate(90, expand=True)
    return img


def extract_team_numbers(
    image_path: Path,
    model: str,
    *,
    client: VisionClient | None = None,
    debug_dir: Path | None = None,
) -> Dict[str, int]:
    vc = client or OpenAIVisionClient()

    img_full = _load_upright(image_path)

    # Try to crop to the boxscore first; full-page photos make the team digits tiny.
    det = detect_boxscore_bbox(img_full, model=model, client=vc)
    if det is not None:
        bbox, rot = det
        if rot == 180:
            img_full = img_full.rotate(180, expand=True)
            det2 = detect_boxscore_bbox(img_full, model=model, client=vc)
            if det2 is not None:
                bbox, _ = det2
        img = img_full.crop(bbox).resize((BASE_W, BASE_H), resample=Image.BILINEAR)
    else:
        img = img_full.resize((BASE_W, BASE_H), resample=Image.BILINEAR)

    w, h = img.size

    # Crop a padded region around each digit and ask the model for the integer.
    # A larger pad helps when photos are framed differently.
    def padded(box_base: Tuple[int, int, int, int], pad: int = 60) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = _scale_box(box_base, w, h)
        return (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))

    # Upscale digits a bit to help legibility.
    home_crop = _img_crop_bytes(img, padded(HOME_TEAM_BOX_BASE), upscale=3)
    visiting_crop = _img_crop_bytes(img, padded(VISITING_TEAM_BOX_BASE), upscale=3)

    prompt = (
        "Read the handwritten team number in this crop. "
        "Return strict JSON: {\"home_team\": <int>, \"visiting_team\": <int>}. "
        "If only one digit is visible, still fill both fields with your best guess."
    )

    # Combine both crops by placing them side-by-side into one image to reduce calls.
    import io
    from PIL import Image as PILImage

    im1 = PILImage.open(io.BytesIO(home_crop))
    im2 = PILImage.open(io.BytesIO(visiting_crop))
    combo = PILImage.new("RGB", (im1.width + im2.width, max(im1.height, im2.height)), (255, 255, 255))
    combo.paste(im1, (0, 0))
    combo.paste(im2, (im1.width, 0))
    bio = io.BytesIO()
    combo.save(bio, format="PNG")

    if debug_dir is not None:
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "teams.png").write_bytes(bio.getvalue())
            img.save(debug_dir / "boxscore_for_teams.png", format="PNG")
        except Exception:
            pass

    data_url = _b64_data_url_bytes(bio.getvalue(), "image/png")

    obj = vision_json(vc, prompt, data_url, model=model, schema=TEAM_SCHEMA)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected teams object, got: {type(obj)}")
    return {"home_team": int(obj["home_team"]), "visiting_team": int(obj["visiting_team"])}


def detect_date_line_bbox(img: Image.Image, model: str, *, client: VisionClient | None = None) -> Tuple[int, int, int, int] | None:
    """Locate the header line containing 'DATE'/'Home Team'/'Visiting Team'/'Hour'.

    Returns pixel bbox or None.
    """

    img_bytes, mime = _img_bytes(img, max_w=1600, fmt="JPEG")
    data_url = _b64_data_url_bytes(img_bytes, mime)

    prompt = (
        "Find the single header line near the top that includes the labels 'DATE', 'Home Team', 'Visiting Team', and 'Hour'. "
        "Return STRICT JSON with normalized coordinates (0..1) for a tight bounding box around that entire line: "
        "{\"x1\":...,\"y1\":...,\"x2\":...,\"y2\":...}."
    )

    try:
        vc = client or OpenAIVisionClient()
        obj = vision_json(vc, prompt, data_url, model=model, schema=DATE_SCHEMA)
        if not isinstance(obj, dict):
            return None
        x1 = max(0.0, min(1.0, float(obj["x1"])))
        y1 = max(0.0, min(1.0, float(obj["y1"])))
        x2 = max(0.0, min(1.0, float(obj["x2"])))
        y2 = max(0.0, min(1.0, float(obj["y2"])))
        if x2 <= x1 or y2 <= y1:
            return None
        W, H = img.size
        return (int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))
    except Exception:
        return None


def detect_boxscore_bbox(
    img: Image.Image,
    model: str,
    *,
    client: VisionClient | None = None,
) -> tuple[tuple[int, int, int, int], int] | None:
    """Try to locate the box-score table region.

    Returns ((x1,y1,x2,y2), rotation) where rotation is 0 or 180.
    Returns None if detection fails.
    """

    # Use a larger max width here because full-page photos make the boxscore table
    # relatively small; downscaling too much hurts detection.
    img_bytes, mime = _img_bytes(img, max_w=2000, fmt="JPEG")
    data_url = _b64_data_url_bytes(img_bytes, mime)

    prompt = (
        "Find the main BOXSCORE table area on this NIL pool league scoresheet photo (HOME TEAM table on left + VISITING TEAM table on right). "
        "Return a TIGHT bounding box that includes the printed headers ('HOME TEAM', 'VISITING TEAM', 'GAME', 'PLAYER', 'TOTAL') and the handwritten score rows. "
        "Do not include the handicap/legend section below the tables unless it is attached to the boxscore region. "
        "Also determine whether the image is upside-down. "
        "Return STRICT JSON with normalized coordinates between 0 and 1 and a rotation field: "
        "{\"x1\":...,\"y1\":...,\"x2\":...,\"y2\":...,\"rotation\":0|180}. "
        "rotation=0 means the text reads normally; rotation=180 means the image is upside-down."
    )

    try:
        vc = client or OpenAIVisionClient()
        obj = vision_json(vc, prompt, data_url, model=model, schema=BOX_SCHEMA)
        if not isinstance(obj, dict):
            return None
        x1 = float(obj["x1"])
        y1 = float(obj["y1"])
        x2 = float(obj["x2"])
        y2 = float(obj["y2"])
        rot = int(obj.get("rotation", 0))
        # clamp
        x1, y1, x2, y2 = [max(0.0, min(1.0, v)) for v in (x1, y1, x2, y2)]
        if x2 <= x1 or y2 <= y1:
            return None
        W, H = img.size
        return ((int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)), rot)
    except Exception:
        return None


def detect_row_bands_by_grid(img_boxscore: Image.Image) -> dict | None:
    """Detect score row bands by locating printed table grid lines.

    Uses OpenCV morphology to isolate the printed horizontal grid lines even in
    the presence of handwriting and shadows.

    Returns normalized coordinates compatible with ROWBANDS_SCHEMA.

    Debug logging includes how many horizontal line candidates we found.
    """

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None

    im = np.array(img_boxscore.convert("L"))
    h, w = im.shape[:2]

    # Adaptive threshold helps with uneven lighting / shadows.
    bw = cv2.adaptiveThreshold(
        im,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )

    # Extract horizontal lines.
    kernel_w = max(20, w // 30)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # Connect broken grid-line segments (handwriting can punch holes in lines).
    horiz = cv2.morphologyEx(
        horiz,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 20), 1)),
        iterations=1,
    )

    # Thicken slightly to make contours easier.
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)), iterations=1)

    contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Candidate y positions for horizontal lines.
    ys: list[int] = []
    min_len = int(w * 0.22)
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww >= min_len and hh <= 18:
            ys.append(y + hh // 2)

    logger.debug(
        "grid_hlines_candidates=%s (min_len=%s) kernel_w=%s",
        len(ys),
        min_len,
        kernel_w,
    )

    if len(ys) < 10:
        return None

    ys.sort()

    # Deduplicate/cluster ys.
    lines: list[int] = []
    cur: list[int] = [ys[0]]
    for y in ys[1:]:
        if abs(y - cur[-1]) <= 3:
            cur.append(y)
        else:
            lines.append(int(sum(cur) / len(cur)))
            cur = [y]
    lines.append(int(sum(cur) / len(cur)))

    # Pick the header separator line: the horizontal line directly under the
    # printed column headers ("Rating | Player | 1 2 3 4 5 6 | Total").
    #
    # Instead of assuming a fixed vertical position, detect it by the pattern of
    # gridline spacing: a relatively short header band followed by a taller score
    # band.
    header: int | None = None
    for i in range(len(lines) - 2):
        y0, y1, y2 = lines[i], lines[i + 1], lines[i + 2]
        dh = y1 - y0
        ds = y2 - y1
        if dh <= 0 or ds <= 0:
            continue

        # Header row is usually modest height; score row below is taller.
        if 18 <= dh <= int(h * 0.10) and ds >= int(dh * 1.25) and ds >= int(h * 0.06):
            header = y1
            break

    # Fallback: best effort (older heuristic)
    if header is None:
        target = int(h * 0.14)
        header = min(lines, key=lambda y: abs(y - target))

    after = [y for y in lines if y > header + 5]
    if len(after) < 6:
        return None

    # Compute bands between consecutive lines.
    bands: list[tuple[int, int, int]] = []
    prev_y = header
    for y in after:
        height = y - prev_y
        if height > 0:
            bands.append((prev_y, y, height))
        prev_y = y

    # Try to pick "score" bands by looking for a repeating 3-band pattern:
    #   SCORE (tall), MARK (shorter), OPPONENTS (shorter).
    # This is more robust than assuming exact triplet alignment.
    min_score_h = int(h * 0.05)

    score_bands: list[tuple[int, int]] = []
    i = 0
    while i + 2 < len(bands) and len(score_bands) < 3:
        a0, b0, h0 = bands[i]
        _, _, h1 = bands[i + 1]
        _, _, h2 = bands[i + 2]

        is_score = (
            h0 >= min_score_h
            and h1 <= int(h0 * 0.85)
            and h2 <= int(h0 * 0.85)
        )

        if is_score:
            score_bands.append((a0, b0))
            i += 3
        else:
            i += 1

    if len(score_bands) < 3:
        return None

    def n(v: int, denom: int) -> float:
        return max(0.0, min(1.0, v / denom))

    # The span between horizontal gridlines may include SCORE plus part of the
    # MARK/OPPONENTS sub-rows (depending on how the sheet is printed).
    # Keep the crop anchored to the top portion where scores are written.
    pad_top = 3
    score_frac = 0.50  # keep top ~50% of the band (exclude mark/opponents rows)
    rows_norm = []
    for (a, b) in score_bands:
        hh = max(1, b - a)
        y1 = a + max(pad_top, int(hh * 0.08))
        y2 = a + int(hh * score_frac)
        if y2 <= y1 + 16:
            y2 = min(b, y1 + 28)
        rows_norm.append({"y1": n(y1, h), "y2": n(y2, h)})

    return {
        "header_y2": n(header, h),
        "home": {"x1": 0.0, "x2": 0.5, "rows": rows_norm},
        "visiting": {"x1": 0.5, "x2": 1.0, "rows": rows_norm},
    }


def detect_row_bands(
    img_boxscore: Image.Image,
    model: str,
    *,
    client: VisionClient | None = None,
) -> dict | None:
    """Detect row bands (score rows only) for home + visiting tables.

    img_boxscore should already be upright and roughly cropped to the boxscore area.
    Returns normalized coordinates in [0,1].
    """

    img_bytes, mime = _img_bytes(img_boxscore, max_w=1280, fmt="JPEG")
    data_url = _b64_data_url_bytes(img_bytes, mime)

    prompt = (
        "This image shows the NIL scoresheet BOXSCORE area with HOME TEAM on the left and VISITING TEAM on the right. "
        "Identify the printed header row that contains labels like 'Rating', 'Player', '1 2 3 4 5 6', and 'Total'. "
        "Return header_y2 as the normalized y-coordinate (0..1) of the bottom of that printed header row. "
        "Then find the 3 handwritten SCORE rows for the HOME TEAM and the 3 handwritten SCORE rows for the VISITING TEAM. "
        "Return STRICT JSON with normalized coordinates between 0 and 1. "
        "For each side, provide x1/x2 that cover rating+player+games 1-6+total, and rows[0..2] as y1/y2 for each SCORE row band only. "
        "IMPORTANT: All score rows must start BELOW header_y2. "
        "Do NOT include the 'mark BR, TR, WZ, WF' row or the 'opponents' row in any band."
    )

    try:
        vc = client or OpenAIVisionClient()
        obj = vision_json(vc, prompt, data_url, model=model, schema=ROWBANDS_SCHEMA)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


class RowExtractError(RuntimeError):
    pass


def extract_rows_by_cropping(
    image_path: Path,
    model: str,
    *,
    client: VisionClient | None = None,
    debug_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    img_full = _load_upright(image_path)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Try to auto-crop to the boxscore table so our fixed boxes are more stable.
    vc = client or OpenAIVisionClient()

    det = detect_boxscore_bbox(img_full, model=model, client=vc)
    if det is not None:
        bbox, rot = det
        if rot == 180:
            img_full = img_full.rotate(180, expand=True)
            det2 = detect_boxscore_bbox(img_full, model=model, client=vc)
            if det2 is not None:
                bbox, _ = det2
        img_box = img_full.crop(bbox)
    else:
        img_box = img_full

    # Normalize to base size for more consistent prompts/debug.
    img_norm = img_box.resize((BASE_W, BASE_H), resample=Image.BILINEAR)
    if debug_dir is not None:
        try:
            img_norm.save(debug_dir / "boxscore.png", format="PNG")
        except Exception:
            pass

    w, h = img_norm.size

    # Anchor-based step: detect exact SCORE row bands inside the boxscore.
    # Use the model first for layout (it can distinguish SCORE vs MARK vs OPPONENTS),
    # then fall back to grid-line detection.
    from layout_detect import detect_scorebands_by_model

    def _layout_is_suspicious(b: dict) -> bool:
        try:
            rows0 = (b.get("home", {}) or {}).get("rows", [])
            if not rows0:
                return True
            y1 = float(rows0[0]["y1"])
            y2 = float(rows0[0]["y2"])
            # If the first "score row" starts too far down, we likely anchored on opponents.
            return y1 > 0.30 or (y2 - y1) < 0.05
        except Exception:
            return True

    bands = detect_scorebands_by_model(img_norm, model=model, client=vc, debug_dir=debug_dir)
    if bands is not None and _layout_is_suspicious(bands):
        logger.debug("rowbands_model_suspicious; ignoring model layout")
        bands = None

    if bands is None:
        bands = detect_row_bands_by_grid(img_norm)
        if bands is None:
            logger.debug("rowbands_grid_failed; falling back to model rowbands")
            bands = detect_row_bands(img_norm, model=model, client=vc)

    if bands is None:
        # Final fallback: use the tuned fixed boxes. This is less robust across
        # framing, but it's better than returning no rows at all.
        def _nxy(v: int, denom: int) -> float:
            return max(0.0, min(1.0, v / float(denom)))

        home_rows = [{"y1": _nxy(y1, BASE_H), "y2": _nxy(y2, BASE_H)} for (_, y1, _, y2) in HOME_ROW_BOXES_BASE]
        vis_rows = [{"y1": _nxy(y1, BASE_H), "y2": _nxy(y2, BASE_H)} for (_, y1, _, y2) in VISITING_ROW_BOXES_BASE]
        bands = {
            "header_y2": _nxy(200, BASE_H),
            "home": {"x1": 0.0, "x2": 0.5, "rows": home_rows},
            "visiting": {"x1": 0.5, "x2": 1.0, "rows": vis_rows},
        }
        logger.debug(
            "rowbands_fallback_fixed header_y2=%.3f rows=%s",
            float(bands.get("header_y2", 0.0)),
            [(r.get("y1"), r.get("y2")) for r in bands.get("home", {}).get("rows", [])],
        )
    else:
        logger.debug(
            "rowbands_ok source=%s header_y2=%.3f rows=%s",
            bands.get("_source", "grid"),
            float(bands.get("header_y2", 0.0)),
            [(r.get("y1"), r.get("y2")) for r in bands.get("home", {}).get("rows", [])],
        )

    if debug_dir is not None and bands is not None:
        try:
            (debug_dir / "rowbands.json").write_text(json.dumps(bands, indent=2), encoding="utf-8")
        except Exception:
            pass

    rows: List[Dict[str, Any]] = []

    def clamp01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    header_y2 = clamp01(bands.get("header_y2", 0.0)) if isinstance(bands, dict) else 0.0

    model_layout = isinstance(bands, dict) and bands.get("_source") == "model"

    def do_side_detected(side: str, band_side: dict, offset: int) -> bool:
        x1n = clamp01(band_side["x1"])
        x2n = clamp01(band_side["x2"])
        last_good_y2n = header_y2

        for idx, r in enumerate(band_side["rows"], start=1):
            y1n = clamp01(r["y1"])
            y2n = clamp01(r["y2"])

            # Enforce monotonic downward movement so later rows can't drift upward
            # into the prior row's mark/opponents area.
            height = max(0.02, y2n - y1n)

            # If row-band detection is slightly off (common on the visiting side),
            # prefer continuity from the last accepted row to avoid starting inside
            # "mark"/"opponents" sub-rows.
            if idx > 1:
                y1n = max(y1n, clamp01(last_good_y2n + height * 0.85))
                y2n = clamp01(y1n + height)

            # Keep rows moving downward, but don't skip too far (otherwise we can
            # land on the printed opponents/mark sub-rows).
            min_y1 = clamp01(last_good_y2n + max(0.005, height * 0.10))
            if y1n < min_y1:
                y1n = min_y1
                y2n = clamp01(y1n + height)

            # Enforce: rows must start below the printed header row.
            if y1n < header_y2:
                dy = (header_y2 - y1n) + 0.02
                y1n = clamp01(y1n + dy)
                y2n = clamp01(y2n + dy)

            # Retry logic: scan downward in fixed steps until we land on the actual
            # player score band (anchored by the small printed row number in the
            # lower-left of the player-name box).
            # Scan around the initially-detected band. Keep the search range
            # tight; a wide scan can accidentally land on adjacent rows and cause
            # row-slot swaps.
            # If we have model-provided layout boxes, they should be close, but can
            # still be slightly tight. Allow a small scan window to improve recall.
            step = 0.006 if model_layout else 0.004
            max_attempts = 5 if model_layout else 3
            player_num = offset + idx

            base_y1n = y1n
            base_y2n = y2n
            up_attempts = 4 if model_layout else 1

            for attempt in range(max_attempts):
                # attempt: 0..max_attempts-1; first `up_attempts` go upward.
                delta = step * (attempt - (up_attempts - 1))
                y1n = clamp01(base_y1n + delta)
                y2n = clamp01(base_y2n + delta)
                logger.info(
                    "DEBUG extract_attempt side=%s player_num=%s attempt=%s",
                    side,
                    player_num,
                    attempt + 1,
                )

                # Add a little padding so we reliably include the full row band,
                # including the tiny printed row index number.
                #
                # Important: don't pad too much vertically; it can bleed into adjacent
                # rows and cause row-slot swaps (e.g., row_index=1 but row2's handwriting).
                base_h = max(0.02, y2n - y1n)
                # Model-detected bands can be very tight; pad more in that case.
                if model_layout:
                    # Model layout is usually close. Pad more on TOP to ensure we keep
                    # the player name + score digits, but pad minimally on the bottom
                    # to avoid bleeding into MARK/OPPONENTS rows.
                    pad = max(0.015, base_h * 0.45)
                else:
                    pad = max(0.006, base_h * 0.15)

                crop_x1n = clamp01(x1n - 0.01)
                crop_x2n = clamp01(x2n + 0.00)
                if model_layout:
                    crop_y1n = clamp01(y1n - pad * 0.50)
                    crop_y2n = clamp01(y2n + pad * 0.12)
                else:
                    crop_y1n = clamp01(y1n - pad * 0.35)
                    crop_y2n = clamp01(y2n + pad * 0.35)

                box = (int(crop_x1n * w), int(crop_y1n * h), int(crop_x2n * w), int(crop_y2n * h))

                # Guard rails: if we hit the bottom of the image, clamp can cause
                # nearly-zero-height crops (e.g. 4px tall). Bail out early.
                if box[3] - box[1] < 24 or box[2] - box[0] < 200:
                    msg = f"Could not process {side} player {player_num} (degenerate crop)"
                    logger.info(
                        "DEBUG extract_failed side=%s player_num=%s attempts=%s (degenerate crop box=%s)",
                        side,
                        player_num,
                        attempt + 1,
                        box,
                    )
                    raise RowExtractError(msg)

                crop = img_norm.crop(box)
                logger.debug(
                    "row_try side=%s idx=%s attempt=%s y1=%.3f y2=%.3f crop_y1=%.3f crop_y2=%.3f px=%s",
                    side,
                    idx,
                    attempt + 1,
                    y1n,
                    y2n,
                    crop_y1n,
                    crop_y2n,
                    box,
                )
                if debug_dir is not None:
                    suffix = "" if attempt == 0 else f"-try{attempt+1}"
                    crop.save(debug_dir / f"{side}-row{idx}{suffix}.png", format="PNG")

                # Pre-check NOTE: previously we tried to detect the two horizontal
                # border lines to avoid half-row crops. In practice this caused missed
                # rows on real photos (glare/shadows). We rely on score sanity checks
                # and tight row-banding instead.

                crop_bytes = _img_crop_bytes(img_norm, box, upscale=2)
                data_url = _b64_data_url_bytes(crop_bytes, "image/png")
                obj = vision_json(vc, ROW_PROMPT, data_url, model=model, schema=ROW_SCHEMA)
                if not isinstance(obj, dict):
                    raise RuntimeError(f"Expected object for {side} row {idx}, got: {type(obj)}")

                player = str(obj.get("player", "")).strip().lower()
                has_opponents_word = bool(obj.get("has_opponents_word", False))
                has_mark_word = bool(obj.get("has_mark_word", False))

                games = [
                    int(obj.get("game1", -1)),
                    int(obj.get("game2", -1)),
                    int(obj.get("game3", -1)),
                    int(obj.get("game4", -1)),
                    int(obj.get("game5", -1)),
                    int(obj.get("game6", -1)),
                ]
                total = int(obj.get("total", -1))
                sum_games = sum(g for g in games if g >= 0)

                invalid_scores = any(v in {8, 9} or v < 0 or (v > 7 and v != 10) for v in games)
                total_mismatch = (not invalid_scores) and (total >= 0) and (sum_games != total)

                # Heuristic: if we hit printed sub-rows, the model tends to output these tokens.
                # Also reject opponent-matchup strings like "1v4" / "4v1".
                import re

                has_alpha = any(ch.isalpha() for ch in player)
                looks_like_name = has_alpha and len(player.strip()) >= 3

                # e.g. "1v4", "4 v 1" (letters other than 'v' are not allowed)
                opponent_like = re.fullmatch(r"\d+\s*v\s*\d+", player) is not None

                hit_keywords = (
                    ("opponent" in player)
                    or ("opponents" in player)
                    or ("mark" in player)
                    or (player in {"wf", "wz", "tr", "br"})
                    or opponent_like
                    or (not has_alpha)
                )

                # "mark" is too flaky (false positives even on clean crops).
                # The model's has_opponents_word is also flaky, so we do NOT hard-reject on it;
                # instead we reject based on player text (e.g., "4v1") and score sanity.

                looks_wrong = (
                    invalid_scores
                    or total_mismatch
                    or hit_keywords
                )

                if looks_wrong:
                    msg = (
                        "row_reject side=%s idx=%s attempt=%s player=%r "
                        "has_opp=%s has_mark=%s invalid_scores=%s total_mismatch=%s hit_keywords=%s games=%s total=%s"
                    )
                    logger.debug(
                        msg,
                        side,
                        idx,
                        attempt + 1,
                        player,
                        has_opponents_word,
                        has_mark_word,
                        invalid_scores,
                        total_mismatch,
                        hit_keywords,
                        games,
                        total,
                    )
                    # Also surface at INFO so it shows up in bot.log.
                    logger.info(
                        "DEBUG " + msg,
                        side,
                        idx,
                        attempt + 1,
                        player,
                        has_opponents_word,
                        has_mark_word,
                        invalid_scores,
                        total_mismatch,
                        hit_keywords,
                        games,
                        total,
                    )
                else:
                    logger.debug(
                        "row_accept side=%s idx=%s attempt=%s player=%r games=%s total=%s",
                        side,
                        idx,
                        attempt + 1,
                        player,
                        games,
                        obj.get("total"),
                    )

                if not looks_wrong:
                    logger.info(
                        "DEBUG extract_found side=%s player_num=%s attempt=%s player=%r",
                        side,
                        player_num,
                        attempt + 1,
                        player,
                    )
                    obj = dict(obj)
                    obj["side"] = side
                    obj["player_num"] = player_num
                    rows.append(obj)
                    last_good_y2n = y2n
                    break

                # Next attempt is controlled by base_y* + delta; no in-loop drift.
            else:
                # Could not find a plausible score row band.
                logger.info(
                    "DEBUG extract_failed side=%s player_num=%s attempts=%s",
                    side,
                    player_num,
                    max_attempts,
                )
                raise RowExtractError(f"Could not process {side} player {player_num}")

        return True

    if isinstance(bands, dict) and "home" in bands and "visiting" in bands and "header_y2" in bands:
        # If home fails, raise immediately (do not continue to visiting).
        do_side_detected("home", bands["home"], offset=0)
        do_side_detected("visiting", bands["visiting"], offset=3)
        return rows

    raise RuntimeError("Row band detection failed (no usable 'home'/'visiting' bands)")


def normalize_rows(image_name: str, extracted: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []

    for i, r in enumerate(extracted, start=1):
        if not isinstance(r, dict):
            raise RuntimeError(f"Row {i} is not an object: {r!r}")

        out: Dict[str, Any] = {}
        for k in [
            "player_num",
            "side",
            "player",
            "game1",
            "game2",
            "game3",
            "game4",
            "game5",
            "game6",
            "total",
        ]:
            if k not in r:
                raise RuntimeError(f"Row {i} missing key '{k}'. Got keys: {sorted(r.keys())}")
            out[k] = r[k]

        out["player_num"] = int(out["player_num"])
        out["side"] = str(out["side"]).strip().lower()
        out["player"] = str(out["player"]).strip()

        for g in ["game1", "game2", "game3", "game4", "game5", "game6", "total"]:
            out[g] = int(out[g])
            # Clamp impossible game values; totals will still be validated separately.
            if g.startswith("game"):
                if out[g] < 0:
                    out[g] = 0
                if out[g] > 10:
                    out[g] = 10

        out_rows.append(out)

    return out_rows


def validate_rows(rows: List[Dict[str, Any]]) -> List[str]:
    warnings: List[str] = []

    if len(rows) != 6:
        warnings.append(f"Expected 6 rows, got {len(rows)}")

    for r in rows:
        # Range checks
        for g in ["game1", "game2", "game3", "game4", "game5", "game6"]:
            v = int(r[g])
            if v < 0 or v > 10:
                warnings.append(f"Out-of-range {g}={v} for {r.get('side')}:{r.get('player')}")

        s = sum(int(r[g]) for g in ["game1", "game2", "game3", "game4", "game5", "game6"])
        if s != int(r["total"]):
            warnings.append(
                f"Total mismatch for P{r.get('player_num')} {r.get('side')}:{r.get('player')}: games sum={s} total={r.get('total')}"
            )

    return warnings


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image",
        default=str(HERE / ".." / "NilSample.jpeg"),
        help="Path to scoresheet image (jpeg/png)",
    )
    ap.add_argument(
        "--model",
        default=os.getenv("BILLIARDSCORES_MODEL", "gpt-4o"),
        help="OpenAI model to use (default: gpt-4o)",
    )
    ap.add_argument(
        "--teams-only",
        action="store_true",
        help="Only extract home_team and visiting_team and print JSON to stdout",
    )
    ap.add_argument(
        "--debug-dir",
        default=None,
        help="Optional directory to write debug crop images (one PNG per row)",
    )
    args = ap.parse_args(argv)

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 2

    try:
        debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else None
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)

        if args.teams_only:
            teams = extract_team_numbers(image_path, model=args.model, client=OpenAIVisionClient(), debug_dir=debug_dir)
            print(json.dumps(teams))
            return 0

        # Debug aid: save the team-number crop + date line crop for the photo.
        if debug_dir is not None:
            try:
                extract_team_numbers(image_path, model=args.model, client=OpenAIVisionClient(), debug_dir=debug_dir)
            except Exception:
                pass

            try:
                img0 = _load_upright(image_path)
                date_bbox = detect_date_line_bbox(img0, model=args.model, client=OpenAIVisionClient())
                if date_bbox is not None:
                    x1, y1, x2, y2 = date_bbox
                    pad = 30
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(img0.width, x2 + pad)
                    y2 = min(img0.height, y2 + pad)
                    date_crop = img0.crop((x1, y1, x2, y2))
                    date_bytes, mime = _img_bytes(date_crop, max_w=1800, fmt="JPEG")
                    (debug_dir / "date.jpg").write_bytes(date_bytes)
            except Exception:
                pass

        extracted = extract_rows_by_cropping(image_path, model=args.model, client=OpenAIVisionClient(), debug_dir=debug_dir)
        rows = normalize_rows(image_path.name, extracted)
        warnings = validate_rows(rows)

        raw_json_path = OUT_DIR / f"{image_path.stem}.extracted.json"
        raw_json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

        out_csv_path = OUT_DIR / f"{image_path.stem}.csv"
        write_csv(rows, out_csv_path)

        print(f"Wrote: {out_csv_path}")
        print(f"Saved raw: {raw_json_path}")
        if warnings:
            print("\nWARNINGS:")
            for w in warnings:
                print(f"- {w}")

        return 0

    except VisionRefusal as e:
        print(
            "ERROR: Vision model refused to extract data from the image. "
            "Try a clearer photo (less glare/shadow) and retry.",
            file=sys.stderr,
        )
        # Keep the raw refusal out of user-facing output; it's in logs/tracebacks.
        return 3

    except Exception as e:
        print(f"ERROR: Extraction failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
