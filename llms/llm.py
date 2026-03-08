from __future__ import annotations

import os
from typing import Any, TypeAlias

from langchain_openai import ChatOpenAI

ChatModel: TypeAlias = ChatOpenAI

_LLM_ROUTER_CONFIG: dict[str, Any] | None = None
_LLM_CACHE: dict[str, ChatModel] = {}

_TASK_ALIASES = {
    "summarize_history": "summarize",
    "decide_retrieval": "decision",
    "rewrite_query": "rewrite",
    "direct_answer": "direct",
    "out_of_scope_answer": "out_of_scope",
    "aggregate_answers": "aggregate",
}


def _build_chat_model(model: str, api_key: str, api_base: str, model_config: dict) -> ChatModel:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=api_base,
        **model_config,
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



def _task_model_map_from_env() -> dict[str, str]:
    mapping = {
        "summarize_history": os.getenv("LLM_MODEL_SUMMARIZE_HISTORY", "").strip(),
        "decide_retrieval": os.getenv("LLM_MODEL_DECIDE_RETRIEVAL", "").strip(),
        "rewrite_query": os.getenv("LLM_MODEL_REWRITE_QUERY", "").strip(),
        "direct_answer": os.getenv("LLM_MODEL_DIRECT_ANSWER", "").strip(),
        "out_of_scope_answer": os.getenv("LLM_MODEL_OUT_OF_SCOPE_ANSWER", "").strip(),
        "research_search": os.getenv("LLM_MODEL_RESEARCH_SEARCH", "").strip(),
        "aggregate_answers": os.getenv("LLM_MODEL_AGGREGATE_ANSWERS", "").strip(),
        "summarize": os.getenv("LLM_MODEL_SUMMARIZE", "").strip(),
        "decision": os.getenv("LLM_MODEL_DECISION", "").strip(),
        "rewrite": os.getenv("LLM_MODEL_REWRITE", "").strip(),
        "direct": os.getenv("LLM_MODEL_DIRECT", "").strip(),
        "out_of_scope": os.getenv("LLM_MODEL_OUT_OF_SCOPE", "").strip(),
        "aggregate": os.getenv("LLM_MODEL_AGGREGATE", "").strip(),
    }
    return {k: v for k, v in mapping.items() if v}



def _resolve_router_config(config: dict | None) -> dict[str, Any]:
    if config is not None:
        cfg = dict(config)
    elif _LLM_ROUTER_CONFIG is not None:
        cfg = dict(_LLM_ROUTER_CONFIG)
    else:
        raise ValueError("LLM router is not configured. Call configure_llm_router() first.")

    existing_task_models = cfg.get("task_models", {}) or {}
    cfg["task_models"] = {**_task_model_map_from_env(), **existing_task_models}
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
        alias = _TASK_ALIASES.get(task_type)
        if alias:
            selected_model = task_models.get(alias)
    if not selected_model:
        selected_model = model

    cache_key = f"{task_type}|{selected_model}|{api_base}|{model_config}"
    if cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    llm = _build_chat_model(selected_model, api_key, api_base, model_config)
    _LLM_CACHE[cache_key] = llm
    return llm
