from __future__ import annotations

import logging
import logging.config
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


_ENV_LINE_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_TASK_MODEL_ENV_NAMES = {
    "summarize_history": "LLM_MODEL_SUMMARIZE_HISTORY",
    "decide_retrieval": "LLM_MODEL_DECIDE_RETRIEVAL",
    "plan_query": "LLM_MODEL_PLAN_QUERY",
    "rewrite_query": "LLM_MODEL_REWRITE_QUERY",
    "direct_answer": "LLM_MODEL_DIRECT_ANSWER",
    "out_of_scope_answer": "LLM_MODEL_OUT_OF_SCOPE_ANSWER",
    "research_search": "LLM_MODEL_RESEARCH_SEARCH",
    "aggregate_answers": "LLM_MODEL_AGGREGATE_ANSWERS",
    "summarize": "LLM_MODEL_SUMMARIZE",
    "decision": "LLM_MODEL_DECISION",
    "rewrite": "LLM_MODEL_REWRITE",
    "direct": "LLM_MODEL_DIRECT",
    "out_of_scope": "LLM_MODEL_OUT_OF_SCOPE",
    "aggregate": "LLM_MODEL_AGGREGATE",
}


def is_offline_mode() -> bool:
    """Check if OFFLINE_MODE environment variable is enabled."""
    return os.getenv("OFFLINE_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def load_dotenv(
    env_path: Path,
    *,
    override: bool = False,
    apply_to_environ: bool = True,
) -> dict[str, str]:
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
        if apply_to_environ and (override or key not in os.environ):
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
    index_root: Path | None = None
    upload_root: Path | None = None
    parsed_artifact_root: Path | None = None
    app_db_path: Path | None = None

    llm_model: str = ""
    llm_api_key: str = ""
    llm_api_base: str = ""
    llm_temperature: float = 0.2
    llm_task_models: dict[str, str] = field(default_factory=dict)

    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_api_base: str = ""
    embedding_provider: str = "openai-compatible"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 20
    embedding_timeout: float | None = None
    embedding_input_mode: str = "raw"
    embedding_check_context_length: bool = False
    embedding_max_input_chars: int = 6000

    vector_backend: str = "faiss"
    lexical_backend: str = "bm25"
    node_backend: str = "json"
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
    retrieval_pipeline: str = "v1_flat_rerank"
    answer_strategy: str = "fixed"

    max_tool_calls: int = 8
    max_iterations: int = 10
    max_context_tokens: int = 5000
    keep_messages: int = 20
    offline_mode: bool = False
    index_write_mode: str = "versioned"
    index_worker_lease_seconds: int = 60
    index_worker_heartbeat_seconds: int = 15
    index_worker_poll_seconds: float = 0.25
    index_worker_max_attempts: int = 3
    upload_max_bytes: int = 50 * 1024 * 1024
    paper_parser: str = "pymupdf4llm"
    parser_timeout_seconds: int = 180
    long_document_timeout_seconds: int = 600

    def __post_init__(self) -> None:
        if self.embedding_input_mode not in {"raw", "tokenized"}:
            raise ValueError("EMBEDDING_INPUT_MODE must be raw or tokenized.")
        if (
            self.embedding_input_mode == "raw"
            and self.embedding_check_context_length
        ):
            raise ValueError(
                "Raw embedding input requires check_embedding_ctx_length=false."
            )
        if self.embedding_dimensions <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be positive.")
        if self.embedding_batch_size <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be positive.")
        if self.embedding_max_input_chars <= 0:
            raise ValueError("EMBEDDING_MAX_INPUT_CHARS must be positive.")
        if self.index_write_mode not in {"versioned", "legacy"}:
            raise ValueError("INDEX_WRITE_MODE must be versioned or legacy.")
        if self.index_worker_lease_seconds <= 0:
            raise ValueError("INDEX_WORKER_LEASE_SECONDS must be positive.")
        if self.index_worker_heartbeat_seconds <= 0:
            raise ValueError("INDEX_WORKER_HEARTBEAT_SECONDS must be positive.")
        if self.index_worker_heartbeat_seconds >= self.index_worker_lease_seconds:
            raise ValueError(
                "INDEX_WORKER_HEARTBEAT_SECONDS must be less than "
                "INDEX_WORKER_LEASE_SECONDS."
            )
        if self.index_worker_poll_seconds <= 0:
            raise ValueError("INDEX_WORKER_POLL_SECONDS must be positive.")
        if self.index_worker_max_attempts <= 0:
            raise ValueError("INDEX_WORKER_MAX_ATTEMPTS must be positive.")
        if self.upload_max_bytes <= 0:
            raise ValueError("UPLOAD_MAX_BYTES must be positive.")
        if self.paper_parser not in {"pymupdf4llm", "legacy"}:
            raise ValueError("PAPER_PARSER must be pymupdf4llm or legacy.")
        if self.answer_strategy not in {"fixed", "adaptive"}:
            raise ValueError("ANSWER_STRATEGY must be fixed or adaptive.")
        from indexing.retrieval_pipeline import get_pipeline_config

        get_pipeline_config(self.retrieval_pipeline)
        if self.parser_timeout_seconds <= 0:
            raise ValueError("PARSER_TIMEOUT_SECONDS must be positive.")
        if self.long_document_timeout_seconds < self.parser_timeout_seconds:
            raise ValueError(
                "LONG_DOCUMENT_TIMEOUT_SECONDS must be greater than or equal to "
                "PARSER_TIMEOUT_SECONDS."
            )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
        self.nodes_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc_trees_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index_root is not None:
            self.index_root.mkdir(parents=True, exist_ok=True)
        if self.upload_root is not None:
            self.upload_root.mkdir(parents=True, exist_ok=True)
        if self.parsed_artifact_root is not None:
            self.parsed_artifact_root.mkdir(parents=True, exist_ok=True)
        if self.app_db_path is not None:
            self.app_db_path.parent.mkdir(parents=True, exist_ok=True)

    def llm_config(self) -> dict:
        return {
            "model": self.llm_model,
            "api_key": self.llm_api_key,
            "api_base": self.llm_api_base,
            "model_config": {"temperature": self.llm_temperature},
            "task_models": dict(self.llm_task_models),
        }

    def indexer_config(self, *, version_dir: Path | None = None) -> dict:
        faiss_dir = version_dir / "faiss" if version_dir else self.faiss_dir
        bm25_path = version_dir / "bm25.pkl" if version_dir else self.bm25_path
        nodes_path = version_dir / "nodes.jsonl" if version_dir else self.nodes_path
        doc_trees_path = (
            version_dir / "doc_trees.json"
            if version_dir
            else self.doc_trees_path
        )
        embedding_cfg: dict[str, object] = {
            "api_key": self.embedding_api_key,
            "api_base": self.embedding_api_base,
            "model": self.embedding_model,
            "provider": self.embedding_provider,
            "dimensions": self.embedding_dimensions,
            "batch_size": self.embedding_batch_size,
            "input_mode": self.embedding_input_mode,
            "check_embedding_ctx_length": self.embedding_check_context_length,
            "max_input_chars": self.embedding_max_input_chars,
        }
        if self.offline_mode:
            embedding_cfg["type"] = "fake"
        if self.embedding_timeout is not None:
            embedding_cfg["timeout"] = self.embedding_timeout

        return {
            "embedding": embedding_cfg,
            "chunker": {"type": self.chunker_type, "params": self.chunker_params},
            "vector_backend": self.vector_backend,
            "lexical_backend": self.lexical_backend,
            "node_backend": self.node_backend,
            "index_mode": self.index_mode,
            "leaf_node_type": self.leaf_node_type,
            "parent_embed_pooling": self.parent_embed_pooling,
            "vectorstore": {"persist_directory": str(faiss_dir)},
            "bm25_path": str(bm25_path),
            "nodes_path": str(nodes_path),
            "doc_trees_path": str(doc_trees_path),
            "retriever": {
                "k": self.retriever_k,
                "alpha": self.fusion_alpha,
                "reranker_backend": self.reranker_backend,
                "flashrank_model": self.flashrank_model,
                "flashrank_cache_dir": self.flashrank_cache_dir,
                "flashrank_top_n": self.flashrank_top_n,
                "pipeline": self.retrieval_pipeline,
            },
        }


def _get_env(
    name: str,
    *aliases: str,
    default: str | None = None,
    env_values: Mapping[str, str] | None = None,
) -> str | None:
    for key in (name, *aliases):
        val = os.environ.get(key)
        if val is None and env_values is not None:
            val = env_values.get(key)
        if val is not None and str(val).strip() != "":
            return val
    return default


def _get_env_int(
    name: str,
    *aliases: str,
    env_values: Mapping[str, str] | None = None,
) -> int | None:
    raw = _get_env(name, *aliases, env_values=env_values)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _get_env_float(
    name: str,
    *aliases: str,
    env_values: Mapping[str, str] | None = None,
) -> float | None:
    raw = _get_env(name, *aliases, env_values=env_values)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


def _get_env_bool(
    name: str,
    *,
    default: bool,
    env_values: Mapping[str, str] | None = None,
) -> bool:
    raw = _get_env(name, env_values=env_values)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


def load_settings(
    *, base_dir: str | Path | None = None, env_file: str | Path | None = None
) -> AppSettings:
    base = (
        Path(base_dir)
        if base_dir is not None
        else Path(__file__).resolve().parent.parent
    )
    env_path = Path(env_file) if env_file is not None else base / ".env"
    env_values = load_dotenv(env_path, apply_to_environ=False)

    def get_env(
        name: str, *aliases: str, default: str | None = None
    ) -> str | None:
        return _get_env(
            name,
            *aliases,
            default=default,
            env_values=env_values,
        )

    def get_env_int(name: str, *aliases: str) -> int | None:
        return _get_env_int(name, *aliases, env_values=env_values)

    def get_env_float(name: str, *aliases: str) -> float | None:
        return _get_env_float(name, *aliases, env_values=env_values)

    data_dir = Path(get_env("DATA_DIR") or (base / "data"))
    index_dir = Path(get_env("INDEX_DIR") or (data_dir / "index"))
    faiss_dir = Path(get_env("FAISS_DIR") or (index_dir / "faiss"))
    bm25_path = Path(get_env("BM25_PATH") or (index_dir / "bm25.pkl"))
    nodes_path = Path(get_env("NODES_PATH") or (index_dir / "nodes.jsonl"))
    doc_trees_path = Path(
        get_env("DOC_TREES_PATH") or (index_dir / "doc_trees.json")
    )
    index_root = Path(get_env("INDEX_ROOT") or (data_dir / "indexes"))
    upload_root = Path(get_env("UPLOAD_ROOT") or (data_dir / "uploads"))
    parsed_artifact_root = Path(
        get_env("PARSED_ARTIFACT_ROOT") or (data_dir / "parsed")
    )
    app_db_path = Path(
        get_env("APP_DB_PATH") or (data_dir / "api" / "sessions.db")
    )

    log_dir = Path(get_env("LOG_DIR") or (base / "logs"))
    log_file = Path(get_env("LOG_FILE") or (log_dir / "agentic_rag.log"))
    log_level = (get_env("LOG_LEVEL", default="INFO") or "INFO").upper()

    llm_model = get_env("LLM_MODEL", "OPENAI_API_MODEL", default="") or ""
    llm_api_key = get_env("OPENAI_API_KEY", default="") or ""
    llm_api_base = get_env("OPENAI_API_BASE", "OPENAI_BASE_URL", default="") or ""
    llm_temperature = get_env_float("LLM_TEMPERATURE")
    llm_task_models = {
        task_type: value
        for task_type, env_name in _TASK_MODEL_ENV_NAMES.items()
        if (value := (get_env(env_name, default="") or "").strip())
    }

    embedding_model = (
        get_env("EMBEDDING_MODEL", default="text-embedding-3-small")
        or "text-embedding-3-small"
    )
    embedding_api_key = (
        get_env("EMBEDDING_API_KEY", "OPENAI_API_KEY", default="") or ""
    )
    embedding_api_base = (
        get_env(
            "EMBEDDING_API_BASE",
            "EMBEDDING_BASE_URL",
            "OPENAI_API_BASE",
            default="",
        )
        or ""
    )
    embedding_provider = (
        get_env("EMBEDDING_PROVIDER", default="openai-compatible")
        or "openai-compatible"
    )
    embedding_dimensions_value = get_env_int(
        "EMBEDDING_DIMENSION",
        "EMBEDDING_DIMENSIONS",
    )
    embedding_dimensions = (
        embedding_dimensions_value
        if embedding_dimensions_value is not None
        else 1536
    )
    embedding_timeout = get_env_float("EMBEDDING_TIMEOUT")
    embedding_batch_size_value = get_env_int("EMBEDDING_BATCH_SIZE")
    embedding_batch_size = (
        embedding_batch_size_value
        if embedding_batch_size_value is not None
        else 20
    )
    embedding_input_mode = (
        get_env("EMBEDDING_INPUT_MODE", default="raw") or "raw"
    ).strip().lower()
    embedding_check_context_length = _get_env_bool(
        "EMBEDDING_CHECK_CONTEXT_LENGTH",
        default=embedding_input_mode != "raw",
        env_values=env_values,
    )
    embedding_max_chars_value = get_env_int("EMBEDDING_MAX_INPUT_CHARS")
    embedding_max_input_chars = (
        embedding_max_chars_value
        if embedding_max_chars_value is not None
        else 6000
    )
    vector_backend = get_env("VECTOR_BACKEND", default="faiss") or "faiss"
    lexical_backend = get_env("LEXICAL_BACKEND", default="bm25") or "bm25"
    node_backend = get_env("NODE_BACKEND", default="json") or "json"

    chunker_type = get_env("CHUNKER_TYPE", default="recursive") or "recursive"
    index_mode = get_env("INDEX_MODE", default="flat") or "flat"
    leaf_node_type = get_env("LEAF_NODE_TYPE", default="paragraph") or "paragraph"
    parent_embed_pooling = (
        get_env("PARENT_EMBED_POOLING", default="mean") or "mean"
    )
    chunk_size = get_env_int("CHUNK_SIZE")
    chunk_overlap = get_env_int("CHUNK_OVERLAP")
    chunker_params: dict[str, object] = {}
    if chunk_size is not None:
        chunker_params["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        chunker_params["chunk_overlap"] = chunk_overlap

    retriever_k = get_env_int("RETRIEVER_K") or 10
    fusion_alpha = get_env_float("FUSION_ALPHA")
    reranker_backend = get_env("RERANKER_BACKEND", default="flashrank") or "flashrank"
    flashrank_model = (
        get_env("FLASHRANK_MODEL", default="ms-marco-TinyBERT-L-2-v2")
        or "ms-marco-TinyBERT-L-2-v2"
    )
    flashrank_cache_dir = get_env("FLASHRANK_CACHE_DIR", default="") or ""
    flashrank_top_n = get_env_int("FLASHRANK_TOP_N") or retriever_k
    retrieval_pipeline = (
        get_env("RETRIEVAL_PIPELINE", default="v1_flat_rerank")
        or "v1_flat_rerank"
    ).strip().lower()
    answer_strategy = (get_env("ANSWER_STRATEGY", default="fixed") or "fixed").strip().lower()

    max_tool_calls = get_env_int("MAX_TOOL_CALLS") or 8
    max_iterations = get_env_int("MAX_ITERATIONS") or 10
    max_context_tokens = get_env_int("MAX_CONTEXT_TOKENS") or 5000
    keep_messages = get_env_int("KEEP_MESSAGES") or 20
    offline_mode = _get_env_bool(
        "OFFLINE_MODE",
        default=False,
        env_values=env_values,
    )
    index_write_mode = (
        get_env("INDEX_WRITE_MODE", default="versioned") or "versioned"
    ).strip().lower()
    lease_seconds_value = get_env_int("INDEX_WORKER_LEASE_SECONDS")
    index_worker_lease_seconds = (
        lease_seconds_value if lease_seconds_value is not None else 60
    )
    heartbeat_seconds_value = get_env_int("INDEX_WORKER_HEARTBEAT_SECONDS")
    index_worker_heartbeat_seconds = (
        heartbeat_seconds_value if heartbeat_seconds_value is not None else 15
    )
    index_worker_poll_seconds = get_env_float("INDEX_WORKER_POLL_SECONDS")
    max_attempts_value = get_env_int("INDEX_WORKER_MAX_ATTEMPTS")
    index_worker_max_attempts = (
        max_attempts_value if max_attempts_value is not None else 3
    )
    upload_max_bytes_value = get_env_int("UPLOAD_MAX_BYTES")
    upload_max_bytes = (
        upload_max_bytes_value
        if upload_max_bytes_value is not None
        else 50 * 1024 * 1024
    )
    paper_parser = (
        get_env("PAPER_PARSER", default="pymupdf4llm") or "pymupdf4llm"
    ).strip().lower()
    parser_timeout_value = get_env_int("PARSER_TIMEOUT_SECONDS")
    parser_timeout_seconds = (
        parser_timeout_value if parser_timeout_value is not None else 180
    )
    long_document_timeout_value = get_env_int("LONG_DOCUMENT_TIMEOUT_SECONDS")
    long_document_timeout_seconds = (
        long_document_timeout_value
        if long_document_timeout_value is not None
        else 600
    )

    settings = AppSettings(
        base_dir=base,
        data_dir=data_dir,
        index_dir=index_dir,
        faiss_dir=faiss_dir,
        bm25_path=bm25_path,
        nodes_path=nodes_path,
        doc_trees_path=doc_trees_path,
        index_root=index_root,
        upload_root=upload_root,
        parsed_artifact_root=parsed_artifact_root,
        app_db_path=app_db_path,
        log_dir=log_dir,
        log_file=log_file,
        log_level=log_level,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_api_base=llm_api_base,
        llm_temperature=llm_temperature if llm_temperature is not None else 0.2,
        llm_task_models=llm_task_models,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_api_base=embedding_api_base,
        embedding_provider=embedding_provider,
        embedding_dimensions=embedding_dimensions,
        embedding_batch_size=embedding_batch_size,
        embedding_timeout=embedding_timeout,
        embedding_input_mode=embedding_input_mode,
        embedding_check_context_length=embedding_check_context_length,
        embedding_max_input_chars=embedding_max_input_chars,
        vector_backend=vector_backend,
        lexical_backend=lexical_backend,
        node_backend=node_backend,
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
        retrieval_pipeline=retrieval_pipeline,
        answer_strategy=answer_strategy,
        max_tool_calls=max_tool_calls,
        max_iterations=max_iterations,
        max_context_tokens=max_context_tokens,
        keep_messages=keep_messages,
        offline_mode=offline_mode,
        index_write_mode=index_write_mode,
        index_worker_lease_seconds=index_worker_lease_seconds,
        index_worker_heartbeat_seconds=index_worker_heartbeat_seconds,
        index_worker_poll_seconds=(
            index_worker_poll_seconds
            if index_worker_poll_seconds is not None
            else 0.25
        ),
        index_worker_max_attempts=index_worker_max_attempts,
        upload_max_bytes=upload_max_bytes,
        paper_parser=paper_parser,
        parser_timeout_seconds=parser_timeout_seconds,
        long_document_timeout_seconds=long_document_timeout_seconds,
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
