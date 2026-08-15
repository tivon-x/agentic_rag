import json
import warnings

import httpx
from langchain_core.messages import HumanMessage

from agent.schemas import GroundedAnswer, RetrievalDecision
from llms.llm import _build_chat_model


def test_dashscope_qwen_function_calling_request_has_thinking_disabled_and_no_parsed_warning() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen3.7-plus",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "RetrievalDecision",
                                        "arguments": json.dumps(
                                            {
                                                "decision": "retrieve",
                                                "reason": "because",
                                            }
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = _build_chat_model(
            "qwen3.7-plus",
            "test-key",
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            {"temperature": 0.2, "http_client": client},
        )
        structured = model.with_config(temperature=0).with_structured_output(
            RetrievalDecision
        )
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            result = structured.invoke([HumanMessage(content="Return JSON")])

    assert result.decision == "retrieve"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["enable_thinking"] is False
    assert payload["tools"]
    assert payload["tool_choice"]
    assert payload["tools"][0]["function"]["strict"] is True
    assert not any("parsed" in str(record.message) for record in records)


def test_non_dashscope_structured_output_keeps_json_schema_default() -> None:
    model = _build_chat_model(
        "gpt-4o-mini",
        "test-key",
        "https://api.openai.com/v1",
        {"temperature": 0.2},
    )

    structured = model.with_structured_output(RetrievalDecision)

    assert (
        structured.first.kwargs["ls_structured_output_format"]["kwargs"]["method"]
        == "json_schema"
    )


def test_dashscope_qwen_explicit_json_mode_keeps_strict_unset() -> None:
    model = _build_chat_model(
        "qwen3.7-plus",
        "test-key",
        "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        {"temperature": 0.2},
    )

    structured = model.with_structured_output(
        RetrievalDecision,
        method="json_mode",
    )

    assert (
        structured.first.kwargs["ls_structured_output_format"]["kwargs"]["method"]
        == "json_mode"
    )


def test_dashscope_qwen_plain_call_does_not_add_structured_overrides() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen3.7-plus",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "plain answer"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = _build_chat_model(
            "qwen3.7-plus",
            "test-key",
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            {"temperature": 0, "http_client": client},
        )
        result = model.invoke([HumanMessage(content="plain")])

    assert result.content == "plain answer"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert "response_format" not in payload
    assert "enable_thinking" not in payload


def test_dashscope_qwen_strict_schema_parses_nested_evidence_list() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        arguments = {
            "answer": "The source supports the claim.",
            "reasoning_summary": "One citation directly supports it.",
            "evidence": [
                {
                    "doc_id": "doc-1",
                    "node_id": "node-1",
                    "source": "paper.pdf",
                    "section_path": ["4", "4.1"],
                    "page": 4,
                    "quote": "The cited passage supports the claim.",
                    "score": 0.91,
                    "relevance": "Direct support.",
                }
            ],
            "confidence": 0.91,
            "limitations": "Limited to the retrieved passage.",
        }
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen3.7-plus",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "GroundedAnswer",
                                        "arguments": json.dumps(arguments),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = _build_chat_model(
            "qwen3.7-plus",
            "test-key",
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            {"temperature": 0, "http_client": client},
        )
        result = model.with_structured_output(GroundedAnswer).invoke(
            [HumanMessage(content="Synthesize the answer.")]
        )

    assert isinstance(result, GroundedAnswer)
    assert isinstance(result.evidence, list)
    assert result.evidence[0].quote == "The cited passage supports the claim."
    assert result.confidence == 0.91
    payload = captured["payload"]
    assert isinstance(payload, dict)
    tool_function = payload["tools"][0]["function"]
    assert tool_function["strict"] is True
    evidence_schema = tool_function["parameters"]["properties"]["evidence"]
    assert evidence_schema["type"] == "array"
    assert evidence_schema["items"]["type"] == "object"
