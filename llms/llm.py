from __future__ import annotations

from typing import Any, Literal, TypeAlias
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI

ChatModel: TypeAlias = ChatOpenAI
_STRICT_UNSET = object()


class _DashScopeQwenChatOpenAI(ChatOpenAI):
    """ChatOpenAI adapter for DashScope Qwen structured outputs.

    DashScope's thinking mode rejects structured-output tool calls unless
    thinking is disabled. Function calling keeps the Pydantic schema in the
    request while ``enable_thinking=False`` keeps the provider compatible.
    """

    def with_structured_output(
        self,
        schema: Any = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = (
            "function_calling"
        ),
        include_raw: bool = False,
        strict: bool | None | object = _STRICT_UNSET,
        tools: list | None = None,
        **kwargs: Any,
    ):
        raw_extra_body = kwargs.pop("extra_body", None)
        extra_body = dict(raw_extra_body or {})
        extra_body["enable_thinking"] = False
        strict_value = (
            True
            if strict is _STRICT_UNSET and method == "function_calling"
            else None
            if strict is _STRICT_UNSET
            else strict
        )
        return super().with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            strict=strict_value,
            tools=tools,
            extra_body=extra_body,
            **kwargs,
        )

_LLM_ROUTER_CONFIG: dict[str, Any] | None = None
_LLM_CACHE: dict[str, ChatModel] = {}


def _build_chat_model(model: str, api_key: str, api_base: str, model_config: dict) -> ChatModel:
    model_type = (
        _DashScopeQwenChatOpenAI
        if _is_dashscope_qwen_model(model, api_base)
        else ChatOpenAI
    )
    return model_type(
        model=model,
        api_key=api_key,
        base_url=api_base,
        **model_config,
    )


def _is_dashscope_qwen_model(model: str, api_base: str) -> bool:
    model_name = model.strip().lower()
    hostname = (urlparse(api_base).hostname or "").lower()
    return model_name.startswith("qwen") and (
        hostname.endswith(".maas.aliyuncs.com")
        or hostname == "dashscope.aliyuncs.com"
    )


def _validate_base_config(config: dict[str, Any]) -> tuple[str, str, str, dict]:
    model = config.get("model", "")
    api_key = config.get("api_key", None)
    api_base = config.get("api_base", None)
    model_config = config.get("model_config", {})

    if not model:
        raise ValueError("Model must be specified in the config.")
    if not api_key:
        raise ValueError("API key must be provided in the config.")
    if not api_base:
        raise ValueError("API base must be provided in the config.")

    return model, api_key, api_base, model_config



def get_llm(config: dict) -> ChatModel | None:
    """Get an LLM instance (OpenAI-compatible mode)."""
    model, api_key, api_base, model_config = _validate_base_config(config)
    return _build_chat_model(model, api_key, api_base, model_config)



def configure_llm_router(config: dict) -> None:
    """Configure global routing config for task-type model selection."""
    global _LLM_ROUTER_CONFIG
    _LLM_ROUTER_CONFIG = dict(config)
    _LLM_CACHE.clear()



def _resolve_router_config(config: dict | None) -> dict[str, Any]:
    if config is not None:
        cfg = dict(config)
    elif _LLM_ROUTER_CONFIG is not None:
        cfg = dict(_LLM_ROUTER_CONFIG)
    else:
        raise ValueError("LLM router is not configured. Call configure_llm_router() first.")

    cfg["task_models"] = dict(cfg.get("task_models", {}) or {})
    return cfg



def get_llm_by_type(task_type: str, config: dict | None = None) -> ChatModel | None:
    """Get an LLM instance based on task type.

    Typical usage in node functions:
    - get_llm_by_type("rewrite_query")

    Optional config override:
    - get_llm_by_type("rewrite_query", config=my_llm_config)
    """
    resolved_cfg = _resolve_router_config(config)
    model, api_key, api_base, model_config = _validate_base_config(resolved_cfg)
    task_models = resolved_cfg.get("task_models", {}) or {}

    selected_model = task_models.get(task_type)
    if not selected_model:
        selected_model = model

    cache_key = f"{task_type}|{selected_model}|{api_base}|{model_config}"
    if cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    llm = _build_chat_model(selected_model, api_key, api_base, model_config)
    _LLM_CACHE[cache_key] = llm
    return llm
