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


ROW_PROMPT = """You are extracting ONE player row from a pool league scoresheet photo.

This is a sports scoresheet (not an ID document). Do NOT identify real people beyond copying the handwritten name as it appears.

Read the row left-to-right and output STRICT JSON with keys:
- player: string (copy the handwritten name as written)
- game1..game6: integers 0-10 (the six GAME columns, in order)
- total: integer

Rules:
- Output MUST be a single JSON object.
- Only include those keys.
- The total should equal sum(game1..game6). If your first read doesn't match, re-check the digits and correct them.
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
    # Row 2 score band (Anthony) — moved down; previous box was hitting opponents row of row1
    (0, 495, 610, 565),
    # Row 3 score band (Ed)
    (0, 675, 610, 745),
]
VISITING_ROW_BOXES_BASE: List[Tuple[int, int, int, int]] = [
    (620, 215, 1280, 285),
    (620, 495, 1280, 565),
    (620, 675, 1280, 745),
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


def openai_vision_json(prompt: str, data_url: str, model: str, schema: dict) -> Any:
    """Call the vision model and force strict JSON.

    We use response_format=json_schema so the model can't reply with prose.
    If the model refuses anyway, we raise VisionRefusal.
    """

    try:
        from openai import OpenAI
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "openai python package not installed. Run: pip install -r requirements.txt"
        ) from e

    client = OpenAI()

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_schema", "json_schema": schema},
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract numbers from sports scoresheets. "
                    "Return only the requested JSON. Do not include commentary."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0,
    )

    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Model returned empty response")

    # If we ever get a refusal/prose response, surface it cleanly.
    lowered = text.lower()
    if "can't assist" in lowered or "cannot assist" in lowered or "i'm sorry" in lowered:
        raise VisionRefusal(text)

    text = _strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model did not return valid JSON. Raw output:\n{text}") from e


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


def extract_team_numbers(image_path: Path, model: str) -> Dict[str, int]:
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
    data_url = _b64_data_url_bytes(bio.getvalue(), "image/png")

    obj = openai_vision_json(prompt, data_url, model=model, schema=TEAM_SCHEMA)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected teams object, got: {type(obj)}")
    return {"home_team": int(obj["home_team"]), "visiting_team": int(obj["visiting_team"])}


def detect_boxscore_bbox(img: Image.Image, model: str) -> tuple[tuple[int, int, int, int], int] | None:
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
        obj = openai_vision_json(prompt, data_url, model=model, schema=BOX_SCHEMA)
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


def extract_rows_by_cropping(
    image_path: Path,
    model: str,
    *,
    debug_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    img_full = _load_upright(image_path)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Try to auto-crop to the boxscore table so our fixed boxes are more stable.
    det = detect_boxscore_bbox(img_full, model=model)
    if det is not None:
        bbox, rot = det
        if rot == 180:
            img_full = img_full.rotate(180, expand=True)
            # bbox was computed on the pre-rotated image; easiest is to rerun detection once.
            det2 = detect_boxscore_bbox(img_full, model=model)
            if det2 is not None:
                bbox, _ = det2
        img = img_full.crop(bbox)
        # Normalize to our base coordinate system so fixed boxes are stable.
        img = img.resize((BASE_W, BASE_H), resample=Image.BILINEAR)
        if debug_dir is not None:
            try:
                img.save(debug_dir / "boxscore.png", format="PNG")
            except Exception:
                pass
    else:
        img = img_full

    w, h = img.size

    rows: List[Dict[str, Any]] = []

    def do_side(side: str, boxes_base: List[Tuple[int, int, int, int]], offset: int):
        for idx, b in enumerate(boxes_base, start=1):
            box = _scale_box(b, w, h)
            crop = img.crop(box)
            if debug_dir is not None:
                crop_path = debug_dir / f"{side}-row{idx}.png"
                crop.save(crop_path, format="PNG")

            crop_bytes = _img_crop_bytes(img, box)
            data_url = _b64_data_url_bytes(crop_bytes, "image/png")
            obj = openai_vision_json(ROW_PROMPT, data_url, model=model, schema=ROW_SCHEMA)
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected object for {side} row {idx}, got: {type(obj)}")
            obj = dict(obj)
            obj["side"] = side
            obj["player_num"] = offset + idx
            rows.append(obj)

    do_side("home", HOME_ROW_BOXES_BASE, offset=0)
    do_side("visiting", VISITING_ROW_BOXES_BASE, offset=3)

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
        if args.teams_only:
            teams = extract_team_numbers(image_path, model=args.model)
            print(json.dumps(teams))
            return 0

        debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else None
        extracted = extract_rows_by_cropping(image_path, model=args.model, debug_dir=debug_dir)
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
