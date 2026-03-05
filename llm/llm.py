from typing import TypeAlias
from langchain_openai import ChatOpenAI

ChatModel: TypeAlias = ChatOpenAI


def get_llm(config: dict) -> ChatModel | None:
    """获取LLM实例（OpenAI兼容模式）"""
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

    return ChatOpenAI(model=model, api_key=api_key, base_url=api_base, **model_config)
