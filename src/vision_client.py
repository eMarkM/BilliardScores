from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol


class VisionClient(Protocol):
    def vision_json(self, *, prompt: str, data_url: str, model: str, schema: dict) -> Any:
        """Return JSON object matching schema."""


@dataclass
class OpenAIVisionClient:
    """Live client using the openai python package."""

    def vision_json(self, *, prompt: str, data_url: str, model: str, schema: dict) -> Any:
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
                    # System messages can only contain text; put the image in a user message.
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the scores from this image."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0,
        )

        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("Model returned empty response")

        lowered = text.lower()
        if "can't assist" in lowered or "cannot assist" in lowered or "i'm sorry" in lowered:
            raise RuntimeError(f"Vision refusal/prose response: {text}")

        # tolerate fenced code blocks
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            t = t.replace("json\n", "", 1).strip()

        try:
            return json.loads(t)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Model did not return valid JSON. Raw output:\n{t}") from e


@dataclass
class CassetteVisionClient:
    """Record/replay client for deterministic tests.

    Keyed by sha256(prompt + schema_name + image_sha256).
    """

    cassette_dir: Path
    mode: str = "replay"  # "replay" | "record"
    live: Optional[VisionClient] = None

    def _key(self, *, prompt: str, data_url: str, schema: dict) -> str:
        schema_name = (schema.get("name") or schema.get("schema", {}).get("title") or "schema")

        # data_url is data:<mime>;base64,<...>
        try:
            b64 = data_url.split(",", 1)[1]
            img_bytes = base64.b64decode(b64)
        except Exception:
            img_bytes = data_url.encode("utf-8")

        img_h = hashlib.sha256(img_bytes).hexdigest()
        base = (prompt + "\n" + str(schema_name) + "\n" + img_h).encode("utf-8")
        return hashlib.sha256(base).hexdigest()

    def vision_json(self, *, prompt: str, data_url: str, model: str, schema: dict) -> Any:
        self.cassette_dir.mkdir(parents=True, exist_ok=True)
        key = self._key(prompt=prompt, data_url=data_url, schema=schema)
        path = self.cassette_dir / f"{key}.json"

        if self.mode == "replay":
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing cassette: {path}. Run with mode='record' to create it."
                )
            return json.loads(path.read_text(encoding="utf-8"))

        if self.mode != "record":
            raise ValueError(f"Unknown CassetteVisionClient.mode: {self.mode}")

        if self.live is None:
            raise RuntimeError("CassetteVisionClient in record mode requires live client")

        obj = self.live.vision_json(prompt=prompt, data_url=data_url, model=model, schema=schema)
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        return obj
