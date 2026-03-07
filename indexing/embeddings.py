"""
Embeddings module containing functions to obtain cloud-based OpenAI-compatible embedding models and get embedding model instances based on configuration.
"""

import hashlib
import os
from dataclasses import dataclass
from typing import Optional
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from langchain_core.embeddings import Embeddings
from core.settings import is_offline_mode


def get_cloud_embeddings(
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs,
) -> OpenAIEmbeddings:
    """
    Get cloud-based OpenAI-compatible embedding model.

    Args:
        model (str): Model name, defaults to "text-embedding-3-small".
        api_key (str): API key.
        api_base (str): API base URL.
        **kwargs: Other parameters passed to OpenAIEmbeddings.

    Returns:
        OpenAIEmbeddings: Returns OpenAI-compatible embedding model instance.
    """
    if not api_key:
        raise ValueError("API key must be provided for cloud embeddings.")
    if not api_base:
        raise ValueError("API base must be provided for cloud embeddings.")

    # langchain_openai expects api_key as SecretStr (or provider callable)
    return OpenAIEmbeddings(
        model=model,
        api_key=SecretStr(api_key),
        base_url=api_base,
        **kwargs,
    )


@dataclass(frozen=True)
class FakeEmbeddings(Embeddings):
    """Deterministic local embeddings for offline/demo mode.

    This avoids any network calls and keeps the rest of the pipeline functional.
    """

    dimensions: int = 384

    def _embed_text(self, text: str) -> list[float]:
        raw = (text or "").encode("utf-8", errors="ignore")
        out: list[float] = []
        counter = 0
        while len(out) < self.dimensions:
            h = hashlib.sha256(raw + counter.to_bytes(4, "little")).digest()
            out.extend([(b / 255.0) for b in h])
            counter += 1
        return out[: self.dimensions]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


def get_embeddings(config: dict) -> Embeddings:
    """
    Get embedding model instance. Prioritizes cloud-based OpenAI-compatible models.

    Args:
        config (dict): Embedding configuration containing optional fields:
            - type: Embedding type, "cloud" or "fake" (defaults to cloud)
            - model: Cloud model name
            - api_key: Cloud API key
            - api_base: Cloud API base URL
            - dimensions: Embedding dimensions (optional)
            - timeout: Request timeout (optional)

    Returns:
        Embeddings: Embedding model instance.
    """
    embedding_config = config.get("embedding", {})

    embedding_type = str(embedding_config.get("type") or "").strip().lower()
    offline = is_offline_mode()

    if offline or embedding_type == "fake":
        dims = embedding_config.get("dimensions")
        dims_int = (
            int(dims) if isinstance(dims, int | str) and str(dims).isdigit() else 384
        )
        return FakeEmbeddings(dimensions=dims_int)

    # Prioritize cloud embeddings
    api_key = (
        embedding_config.get("api_key")
        or os.getenv("EMBEDDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    api_base = (
        embedding_config.get("api_base")
        or os.getenv("EMBEDDING_API_BASE")
        or os.getenv("OPENAI_API_BASE")
    )
    model = embedding_config.get(
        "model", os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # If api_key and api_base are configured, use cloud embeddings
    if api_key and api_base:
        return get_cloud_embeddings(
            model=model,
            api_key=api_key,
            api_base=api_base,
            # Support additional parameters
            dimensions=embedding_config.get("dimensions"),
            timeout=embedding_config.get("timeout"),
        )

    else:
        raise ValueError(
            "No valid embeddings configuration found. Provide api_key + api_base, or set OFFLINE_MODE=1 to use FakeEmbeddings."
        )
