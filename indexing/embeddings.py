"""Embedding providers and application-side input contract enforcement."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr


def get_cloud_embeddings(
    model: str = "text-embedding-3-small",
    api_key: str | None = None,
    api_base: str | None = None,
    *,
    check_embedding_ctx_length: bool = False,
    **kwargs: Any,
) -> OpenAIEmbeddings:
    """Build an OpenAI-compatible embedding client."""
    if not api_key:
        raise ValueError("API key must be provided for cloud embeddings.")
    if not api_base:
        raise ValueError("API base must be provided for cloud embeddings.")

    provider_kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return OpenAIEmbeddings(
        model=model,
        api_key=SecretStr(api_key),
        base_url=api_base,
        check_embedding_ctx_length=check_embedding_ctx_length,
        **provider_kwargs,
    )


@dataclass(frozen=True)
class FakeEmbeddings(Embeddings):
    """Deterministic local embeddings for offline and test runs."""

    dimensions: int = 384

    def _embed_text(self, text: str) -> list[float]:
        raw = (text or "").encode("utf-8", errors="ignore")
        output: list[float] = []
        counter = 0
        while len(output) < self.dimensions:
            digest = hashlib.sha256(raw + counter.to_bytes(4, "little")).digest()
            output.extend(byte / 255.0 for byte in digest)
            counter += 1
        return output[: self.dimensions]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


@dataclass(frozen=True)
class LengthLimitedEmbeddings(Embeddings):
    """Reject inputs that exceed the configured application-side contract."""

    delegate: Embeddings
    max_input_chars: int

    def _validate(self, text: str) -> None:
        if len(text) > self.max_input_chars:
            raise ValueError(
                "Embedding input exceeds EMBEDDING_MAX_INPUT_CHARS="
                f"{self.max_input_chars}; split the passage before indexing."
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            self._validate(text)
        return self.delegate.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self._validate(text)
        return self.delegate.embed_query(text)


def get_embeddings(config: dict[str, Any]) -> Embeddings:
    """Build an embedding model exclusively from the supplied settings config."""
    embedding_config = config.get("embedding", {})
    embedding_type = str(embedding_config.get("type") or "cloud").strip().lower()
    dimensions = int(embedding_config.get("dimensions") or 1536)
    max_input_chars = int(embedding_config.get("max_input_chars") or 6000)

    if embedding_type == "fake":
        delegate: Embeddings = FakeEmbeddings(dimensions=dimensions)
    else:
        provider = str(
            embedding_config.get("provider") or "openai-compatible"
        ).strip().lower()
        if provider not in {"openai-compatible", "openai"}:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        input_mode = str(embedding_config.get("input_mode") or "raw").strip().lower()
        check_context_length = bool(
            embedding_config.get("check_embedding_ctx_length", input_mode != "raw")
        )
        if input_mode == "raw" and check_context_length:
            raise ValueError(
                "Raw embedding input requires check_embedding_ctx_length=false."
            )
        delegate = get_cloud_embeddings(
            model=str(
                embedding_config.get("model") or "text-embedding-3-small"
            ),
            api_key=str(embedding_config.get("api_key") or ""),
            api_base=str(embedding_config.get("api_base") or ""),
            check_embedding_ctx_length=check_context_length,
            dimensions=dimensions,
            chunk_size=int(embedding_config.get("batch_size") or 20),
            timeout=embedding_config.get("timeout"),
        )

    return LengthLimitedEmbeddings(
        delegate=delegate,
        max_input_chars=max_input_chars,
    )
