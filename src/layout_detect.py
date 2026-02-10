from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from extract_nil_sample_to_csv import _b64_data_url_bytes, _img_bytes, vision_json
from vision_client import OpenAIVisionClient, VisionClient


SCOREBANDS_SCHEMA: dict[str, Any] = {
    "name": "nil_scoresheet_scorebands",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "header_y2": {
                "type": "number",
                "description": "Normalized y (0..1) for the bottom of the printed column header row (Rating/Player/1..6/Total).",
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
                            "properties": {"y1": {"type": "number"}, "y2": {"type": "number"}},
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
                            "properties": {"y1": {"type": "number"}, "y2": {"type": "number"}},
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


def detect_scorebands_by_model(
    img_boxscore: Image.Image,
    *,
    model: str,
    client: VisionClient | None = None,
    debug_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Ask the vision model to return tight score-row bands.

    This is intentionally *layout-only*: no score reading.

    Returns a dict compatible with the existing bands shape:
    {header_y2, home:{x1,x2,rows:[{y1,y2}...]}, visiting:{...}}
    plus an internal _source='model'.
    """

    vc = client or OpenAIVisionClient()

    img_bytes, mime = _img_bytes(img_boxscore, max_w=1600, fmt="JPEG")
    data_url = _b64_data_url_bytes(img_bytes, mime)

    prompt = (
        "This image shows the NIL scoresheet BOXSCORE area with HOME TEAM table on the left and VISITING TEAM table on the right. "
        "Find the printed column header row that reads like 'Rating | Player | 1 | 2 | 3 | 4 | 5 | 6 | Total' and return header_y2 as the bottom of that header row. "
        "Then locate the THREE player SCORE rows for each side (HOME and VISITING).\n\n"
        "IMPORTANT: Each player block contains three sub-rows: a SCORE row (with the handwritten scores), a MARK row (with 'mark BR, TR, WZ, WF'), and an OPPONENTS row (with the word 'opponents' and matchup strings like '1v4').\n"
        "Return y1/y2 for the SCORE row ONLY (exclude MARK and OPPONENTS).\n"
        "- Each returned SCORE band MUST include the handwritten player name and multiple handwritten score digits.\n"
        "- Each returned SCORE band MUST NOT include the printed word 'opponents' or any matchup strings like '1v4'.\n"
        "- Make each y-band tall enough to read handwriting (prefer y2-y1 >= 0.07).\n\n"
        "Return STRICT JSON with normalized coordinates (0..1). Provide home.x1/x2 and visiting.x1/x2 covering rating+player+1..6+total. "
        "All rows must be below header_y2."
    )

    try:
        obj = vision_json(vc, prompt, data_url, model=model, schema=SCOREBANDS_SCHEMA)
        if not isinstance(obj, dict):
            return None
        obj = dict(obj)
        obj["_source"] = "model"

        if debug_dir is not None:
            try:
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / "scorebands_model.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")
            except Exception:
                pass

        return obj
    except Exception:
        return None
