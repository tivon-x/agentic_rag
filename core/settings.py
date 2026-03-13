from __future__ import annotations

import logging
import logging.config
import os
import re
from dataclasses import dataclass, field
from pathlib import Path


_ENV_LINE_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def is_offline_mode() -> bool:
    """Check if OFFLINE_MODE environment variable is enabled."""
    return os.getenv("OFFLINE_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def load_dotenv(env_path: Path, *, override: bool = False) -> dict[str, str]:
    """Minimal .env loader (no external dependency).

    Supports lines like:
    - KEY=VALUE
    - export KEY=VALUE
    - quoted values with '...' or "..."
    """
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE_RE.match(line)
        if not m:
            continue
        key, raw_value = m.group(1), m.group(2).strip()
        if (
            len(raw_value) >= 2
            and raw_value[0] in {'"', "'"}
            and raw_value[-1] == raw_value[0]
        ):
            value = raw_value[1:-1]
        else:
            value = raw_value

        values[key] = value
        if override or key not in os.environ:
            os.environ[key] = value

    return values


@dataclass(frozen=True)
class AppSettings:
    base_dir: Path

    data_dir: Path
    index_dir: Path
    faiss_dir: Path
    bm25_path: Path
    nodes_path: Path
    doc_trees_path: Path

    log_dir: Path
    log_file: Path
    log_level: str = "INFO"

    llm_model: str = ""
    llm_api_key: str = ""
    llm_api_base: str = ""
    llm_temperature: float = 0.2

    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_api_base: str = ""
    embedding_dimensions: int | None = None
    embedding_timeout: float | None = None

    chunker_type: str = "recursive"
    chunker_params: dict[str, object] = field(default_factory=dict)
    index_mode: str = "flat"
    leaf_node_type: str = "paragraph"
    parent_embed_pooling: str = "mean"

    retriever_k: int = 10
    fusion_alpha: float = 0.5
    reranker_backend: str = "flashrank"
    flashrank_model: str = "ms-marco-TinyBERT-L-2-v2"
    flashrank_cache_dir: str = ""
    flashrank_top_n: int = 10

    max_tool_calls: int = 8
    max_iterations: int = 10
    max_context_tokens: int = 5000
    keep_messages: int = 20
    offline_mode: bool = False

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
        self.nodes_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc_trees_path.parent.mkdir(parents=True, exist_ok=True)

    def llm_config(self) -> dict:
        return {
            "model": self.llm_model,
            "api_key": self.llm_api_key,
            "api_base": self.llm_api_base,
            "model_config": {"temperature": self.llm_temperature},
        }

    def indexer_config(self) -> dict:
        embedding_cfg: dict[str, object] = {
            "api_key": self.embedding_api_key,
            "api_base": self.embedding_api_base,
            "model": self.embedding_model,
        }
        if self.embedding_dimensions is not None:
            embedding_cfg["dimensions"] = self.embedding_dimensions
        if self.embedding_timeout is not None:
            embedding_cfg["timeout"] = self.embedding_timeout

        return {
            "embedding": embedding_cfg,
            "chunker": {"type": self.chunker_type, "params": self.chunker_params},
            "index_mode": self.index_mode,
            "leaf_node_type": self.leaf_node_type,
            "parent_embed_pooling": self.parent_embed_pooling,
            "vectorstore": {"persist_directory": str(self.faiss_dir)},
            "bm25_path": str(self.bm25_path),
            "nodes_path": str(self.nodes_path),
            "doc_trees_path": str(self.doc_trees_path),
            "retriever": {
                "k": self.retriever_k,
                "alpha": self.fusion_alpha,
                "reranker_backend": self.reranker_backend,
                "flashrank_model": self.flashrank_model,
                "flashrank_cache_dir": self.flashrank_cache_dir,
                "flashrank_top_n": self.flashrank_top_n,
            },
        }


def _get_env(name: str, *aliases: str, default: str | None = None) -> str | None:
    for key in (name, *aliases):
        val = os.getenv(key)
        if val is not None and str(val).strip() != "":
            return val
    return default


def _get_env_int(name: str, *aliases: str) -> int | None:
    raw = _get_env(name, *aliases)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _get_env_float(name: str, *aliases: str) -> float | None:
    raw = _get_env(name, *aliases)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load_settings(
    *, base_dir: str | Path | None = None, env_file: str | Path | None = None
) -> AppSettings:
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    env_path = Path(env_file) if env_file is not None else base / ".env"
    load_dotenv(env_path, override=False)

    data_dir = Path(_get_env("DATA_DIR") or (base / "data"))
    index_dir = Path(_get_env("INDEX_DIR") or (data_dir / "index"))
    faiss_dir = Path(_get_env("FAISS_DIR") or (index_dir / "faiss"))
    bm25_path = Path(_get_env("BM25_PATH") or (index_dir / "bm25.pkl"))
    nodes_path = Path(_get_env("NODES_PATH") or (index_dir / "nodes.jsonl"))
    doc_trees_path = Path(
        _get_env("DOC_TREES_PATH") or (index_dir / "doc_trees.json")
    )

    log_dir = Path(_get_env("LOG_DIR") or (base / "logs"))
    log_file = Path(_get_env("LOG_FILE") or (log_dir / "agentic_rag.log"))
    log_level = (_get_env("LOG_LEVEL", default="INFO") or "INFO").upper()

    llm_model = _get_env("LLM_MODEL", "OPENAI_API_MODEL", default="") or ""
    llm_api_key = _get_env("OPENAI_API_KEY", default="") or ""
    llm_api_base = _get_env("OPENAI_API_BASE", "OPENAI_BASE_URL", default="") or ""
    llm_temperature = _get_env_float("LLM_TEMPERATURE")

    embedding_model = (
        _get_env("EMBEDDING_MODEL", default="text-embedding-3-small")
        or "text-embedding-3-small"
    )
    embedding_api_key = (
        _get_env("EMBEDDING_API_KEY", "OPENAI_API_KEY", default="") or ""
    )
    embedding_api_base = (
        _get_env(
            "EMBEDDING_API_BASE",
            "EMBEDDING_BASE_URL",
            "OPENAI_API_BASE",
            default="",
        )
        or ""
    )
    embedding_dimensions = _get_env_int("EMBEDDING_DIMENSION", "EMBEDDING_DIMENSIONS")
    embedding_timeout = _get_env_float("EMBEDDING_TIMEOUT")

    chunker_type = _get_env("CHUNKER_TYPE", default="recursive") or "recursive"
    index_mode = _get_env("INDEX_MODE", default="flat") or "flat"
    leaf_node_type = _get_env("LEAF_NODE_TYPE", default="paragraph") or "paragraph"
    parent_embed_pooling = (
        _get_env("PARENT_EMBED_POOLING", default="mean") or "mean"
    )
    chunk_size = _get_env_int("CHUNK_SIZE")
    chunk_overlap = _get_env_int("CHUNK_OVERLAP")
    chunker_params: dict[str, object] = {}
    if chunk_size is not None:
        chunker_params["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        chunker_params["chunk_overlap"] = chunk_overlap

    retriever_k = _get_env_int("RETRIEVER_K") or 10
    fusion_alpha = _get_env_float("FUSION_ALPHA")
    reranker_backend = _get_env("RERANKER_BACKEND", default="flashrank") or "flashrank"
    flashrank_model = (
        _get_env("FLASHRANK_MODEL", default="ms-marco-TinyBERT-L-2-v2")
        or "ms-marco-TinyBERT-L-2-v2"
    )
    flashrank_cache_dir = _get_env("FLASHRANK_CACHE_DIR", default="") or ""
    flashrank_top_n = _get_env_int("FLASHRANK_TOP_N") or retriever_k

    max_tool_calls = _get_env_int("MAX_TOOL_CALLS") or 8
    max_iterations = _get_env_int("MAX_ITERATIONS") or 10
    max_context_tokens = _get_env_int("MAX_CONTEXT_TOKENS") or 5000
    keep_messages = _get_env_int("KEEP_MESSAGES") or 20
    offline_mode = is_offline_mode()

    settings = AppSettings(
        base_dir=base,
        data_dir=data_dir,
        index_dir=index_dir,
        faiss_dir=faiss_dir,
        bm25_path=bm25_path,
        nodes_path=nodes_path,
        doc_trees_path=doc_trees_path,
        log_dir=log_dir,
        log_file=log_file,
        log_level=log_level,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_api_base=llm_api_base,
        llm_temperature=llm_temperature if llm_temperature is not None else 0.2,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_api_base=embedding_api_base,
        embedding_dimensions=embedding_dimensions,
        embedding_timeout=embedding_timeout,
        chunker_type=chunker_type,
        chunker_params=chunker_params,
        index_mode=index_mode,
        leaf_node_type=leaf_node_type,
        parent_embed_pooling=parent_embed_pooling,
        retriever_k=retriever_k,
        fusion_alpha=fusion_alpha if fusion_alpha is not None else 0.5,
        reranker_backend=reranker_backend,
        flashrank_model=flashrank_model,
        flashrank_cache_dir=flashrank_cache_dir,
        flashrank_top_n=flashrank_top_n,
        max_tool_calls=max_tool_calls,
        max_iterations=max_iterations,
        max_context_tokens=max_context_tokens,
        keep_messages=keep_messages,
        offline_mode=offline_mode,
    )
    settings.ensure_dirs()
    return settings


def configure_logging(settings: AppSettings) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    level = settings.log_level.upper()
    if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        level = "INFO"

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": level,
                    "formatter": "standard",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "level": level,
                    "formatter": "standard",
                    "filename": str(settings.log_file),
                    "encoding": "utf-8",
                },
            },
            "root": {
                "level": level,
                "handlers": ["console", "file"],
            },
        }
    )
