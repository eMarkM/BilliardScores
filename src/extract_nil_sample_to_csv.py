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
import os
import sys
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageOps

# Optional: help type-checkers; not required at runtime.


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

IMPORTANT: This crop may also include the TOTAL box at the far right.
- game1..game6 are the SIX game columns labeled 1,2,3,4,5,6
- total is the separate TOTAL column at the far right
- Do NOT copy the total into game6.

Read the score row left-to-right and output STRICT JSON with keys:
- player: string (copy the handwritten name as written)
- game1..game6: integers 0-7 or 10 ONLY (the six GAME columns, in order)
- total: integer (TOTAL column)

Rules:
- Output MUST be a single JSON object.
- Only include those keys.
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
    img = _load_upright(image_path)
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
        except Exception:
            pass

    data_url = _b64_data_url_bytes(bio.getvalue(), "image/png")

    vc = client or OpenAIVisionClient()
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

    img_bytes, mime = _img_bytes(img, max_w=1280, fmt="JPEG")
    data_url = _b64_data_url_bytes(img_bytes, mime)

    prompt = (
        "Find the main box-score table area on this NIL pool league scoresheet photo. "
        "Also determine whether the image is upside-down. "
        "Return STRICT JSON with normalized coordinates between 0 and 1 and a rotation field: "
        "{\"x1\":...,\"y1\":...,\"x2\":...,\"y2\":...,\"rotation\":0|180}. "
        "rotation=0 means the text reads normally; rotation=180 means the image is upside-down. "
        "The box should include the player name column(s) and game score columns for both teams."
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

    # Anchor-based step: detect exact score-row bands inside the boxscore.
    bands = detect_row_bands(img_norm, model=model, client=vc)
    if debug_dir is not None and bands is not None:
        try:
            (debug_dir / "rowbands.json").write_text(json.dumps(bands, indent=2), encoding="utf-8")
        except Exception:
            pass

    rows: List[Dict[str, Any]] = []

    def clamp01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    header_y2 = clamp01(bands.get("header_y2", 0.0)) if isinstance(bands, dict) else 0.0

    def do_side_detected(side: str, band_side: dict, offset: int):
        x1n = clamp01(band_side["x1"])
        x2n = clamp01(band_side["x2"])
        for idx, r in enumerate(band_side["rows"], start=1):
            y1n = clamp01(r["y1"])
            y2n = clamp01(r["y2"])

            # Enforce: rows must start below the printed header row.
            if y1n < header_y2:
                dy = (header_y2 - y1n) + 0.02
                y1n = clamp01(y1n + dy)
                y2n = clamp01(y2n + dy)

            # Retry logic: if a band lands on "opponents"/"mark" rows, nudge downward.
            step = 0.12
            for attempt in range(3):
                box = (int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h))
                crop = img_norm.crop(box)
                if debug_dir is not None:
                    suffix = "" if attempt == 0 else f"-try{attempt+1}"
                    crop.save(debug_dir / f"{side}-row{idx}{suffix}.png", format="PNG")

                crop_bytes = _img_crop_bytes(img_norm, box)
                data_url = _b64_data_url_bytes(crop_bytes, "image/png")
                obj = vision_json(vc, ROW_PROMPT, data_url, model=model, schema=ROW_SCHEMA)
                if not isinstance(obj, dict):
                    raise RuntimeError(f"Expected object for {side} row {idx}, got: {type(obj)}")

                player = str(obj.get("player", "")).strip().lower()

                games = [
                    int(obj.get("game1", -1)),
                    int(obj.get("game2", -1)),
                    int(obj.get("game3", -1)),
                    int(obj.get("game4", -1)),
                    int(obj.get("game5", -1)),
                    int(obj.get("game6", -1)),
                ]
                invalid_scores = any(v in {8, 9} or v < 0 or (v > 7 and v != 10) for v in games)

                # Heuristic: if we hit printed sub-rows, the model tends to output these tokens.
                looks_wrong = (
                    invalid_scores
                    or ("opponent" in player)
                    or ("opponents" in player)
                    or (player in {"wf", "wz", "tr", "br"})
                )
                if not looks_wrong:
                    obj = dict(obj)
                    obj["side"] = side
                    obj["player_num"] = offset + idx
                    rows.append(obj)
                    break

                # Nudge downward and retry
                y1n = clamp01(y1n + step)
                y2n = clamp01(y2n + step)
            else:
                # If we exhausted retries, accept the last extraction result.
                obj = dict(obj)
                obj["side"] = side
                obj["player_num"] = offset + idx
                rows.append(obj)

    if isinstance(bands, dict) and "home" in bands and "visiting" in bands and "header_y2" in bands:
        do_side_detected("home", bands["home"], offset=0)
        do_side_detected("visiting", bands["visiting"], offset=3)
        return rows

    # Fallback to legacy fixed boxes.
    def do_side_fixed(side: str, boxes_base: List[Tuple[int, int, int, int]], offset: int):
        for idx, b in enumerate(boxes_base, start=1):
            box = b  # already in BASE_W/BASE_H coords because img_norm is normalized
            crop = img_norm.crop(box)
            if debug_dir is not None:
                crop.save(debug_dir / f"{side}-row{idx}.png", format="PNG")

            crop_bytes = _img_crop_bytes(img_norm, box)
            data_url = _b64_data_url_bytes(crop_bytes, "image/png")
            obj = vision_json(vc, ROW_PROMPT, data_url, model=model, schema=ROW_SCHEMA)
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected object for {side} row {idx}, got: {type(obj)}")
            obj = dict(obj)
            obj["side"] = side
            obj["player_num"] = offset + idx
            rows.append(obj)

    do_side_fixed("home", HOME_ROW_BOXES_BASE, offset=0)
    do_side_fixed("visiting", VISITING_ROW_BOXES_BASE, offset=3)
    return rows


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
