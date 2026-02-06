from __future__ import annotations

import types

from vision_client import OpenAIVisionClient


def test_openai_vision_client_sends_image_in_user_role(monkeypatch):
    captured: dict = {}

    class FakeChatCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)

            # Minimal response shape used by OpenAIVisionClient
            msg = types.SimpleNamespace(content="{\"ok\": true}")
            choice = types.SimpleNamespace(message=msg)
            return types.SimpleNamespace(choices=[choice])

    class FakeChat:
        def __init__(self):
            self.completions = FakeChatCompletions()

    class FakeOpenAI:
        def __init__(self):
            self.chat = FakeChat()

    # OpenAIVisionClient imports `OpenAI` from the `openai` module at call-time.
    # Provide a fake `openai` module in sys.modules so the import succeeds.
    import sys

    fake_openai_mod = types.SimpleNamespace(OpenAI=FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_mod)

    client = OpenAIVisionClient()
    obj = client.vision_json(
        prompt="SYSTEM PROMPT",
        data_url="data:image/jpeg;base64,Zm9v",  # 'foo'
        model="gpt-4o",
        schema={"name": "x", "schema": {"type": "object"}},
    )

    assert obj == {"ok": True}

    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert isinstance(messages[0]["content"], str)

    assert messages[1]["role"] == "user"
    content = messages[1]["content"]

    assert any(c.get("type") == "image_url" for c in content)
