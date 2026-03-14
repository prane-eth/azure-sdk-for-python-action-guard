import json
import pytest

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference._action_guard import GuardDecision
from azure.ai.inference import models as _models
import azure.ai.inference._patch as _patch_module
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import ChatRequestMessage


class DummyResponse:
    def __init__(self, choices):
        self.choices = choices


class DummyChoice:
    def __init__(self, message=None):
        self.message = message


class DummyToolCall:
    def __init__(self, id, function_name, arguments):
        self.id = id
        self.function = type(
            "F", (), {"name": function_name, "arguments": json.dumps(arguments)}
        )


class DummyMessage:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls


def test_action_guard_blocks_non_streaming(monkeypatch):
    client = ChatCompletionsClient("https://example", AzureKeyCredential("key"))

    # Simulate a server response object with a tool call
    tool_call = DummyToolCall("1", "dangerous", {"cmd": "rm -rf /"})
    message = DummyMessage(tool_calls=[tool_call])
    choice = DummyChoice(message=message)
    fake_result = type("R", (), {"choices": [choice]})

    # Monkeypatch the module-level deserializer to return our fake result
    monkeypatch.setattr(
        _patch_module, "_deserialize", lambda *args, **kwargs: fake_result, raising=True
    )

    def guard(call):
        assert call is tool_call or getattr(call, "id", None) == "1"
        return GuardDecision.BLOCK

    with pytest.raises(ValueError, match="Tool call blocked by action_guard"):
        # call complete which will hit our patched deserialize path
        client.complete(
            messages=[ChatRequestMessage(role="user", content="hi")], action_guard=guard
        )


def test_action_guard_allows_non_streaming(monkeypatch):
    client = ChatCompletionsClient("https://example", AzureKeyCredential("key"))

    tool_call = DummyToolCall("1", "safe", {"q": "ok"})
    message = DummyMessage(tool_calls=[tool_call])
    choice = DummyChoice(message=message)
    fake_result = type("R", (), {"choices": [choice]})

    monkeypatch.setattr(
        _patch_module, "_deserialize", lambda *args, **kwargs: fake_result, raising=True
    )

    def guard(call):
        return GuardDecision.ALLOW

    res = client.complete(
        messages=[ChatRequestMessage(role="user", content="hi")], action_guard=guard
    )
    assert res is fake_result


def test_action_guard_blocks_streaming(monkeypatch):
    # For streaming we simulate StreamingChatCompletions iterator
    client = ChatCompletionsClient("https://example", AzureKeyCredential("key"))

    tool_call = DummyToolCall("s1", "dangerous", {"cmd": "rm -rf /"})
    update = type(
        "U",
        (),
        {
            "choices": [
                type("C", (), {"delta": type("D", (), {"tool_calls": [tool_call]})()})()
            ]
        },
    )

    class FakeStreaming:
        def __init__(self):
            self._iter = iter([update])
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._iter)

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        _models,
        "StreamingChatCompletions",
        lambda response: FakeStreaming(),
        raising=True,
    )

    def guard(call):
        return GuardDecision.BLOCK

    gen = client.complete(
        messages=[ChatRequestMessage(role="user", content="hi")],
        stream=True,
        action_guard=guard,
    )
    with pytest.raises(ValueError, match="Tool call blocked by action_guard"):
        list(gen)


def test_action_guard_allows_streaming(monkeypatch):
    client = ChatCompletionsClient("https://example", AzureKeyCredential("key"))

    tool_call = DummyToolCall("s1", "safe", {"x": 1})
    update = type(
        "U",
        (),
        {
            "choices": [
                type("C", (), {"delta": type("D", (), {"tool_calls": [tool_call]})()})()
            ]
        },
    )

    class FakeStreaming:
        def __init__(self):
            self._iter = iter([update])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._iter)

        def close(self):
            pass

    monkeypatch.setattr(
        _models,
        "StreamingChatCompletions",
        lambda response: FakeStreaming(),
        raising=True,
    )

    def guard(call):
        return GuardDecision.ALLOW

    gen = client.complete(
        messages=[ChatRequestMessage(role="user", content="hi")],
        stream=True,
        action_guard=guard,
    )
    items = list(gen)
    assert items and items[0] is update
