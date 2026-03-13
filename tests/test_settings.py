"""Tests for settings module."""

import os
from core.settings import (
    load_settings,
    is_offline_mode,
    load_dotenv,
    configure_logging,
)


def test_load_settings_defaults(tmp_path, monkeypatch):
    """Test load_settings with default values (no .env file)."""
    # Clear environment to ensure defaults
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OFFLINE_MODE", raising=False)

    settings = load_settings(base_dir=tmp_path, env_file=tmp_path / "nonexistent.env")

    assert settings.base_dir == tmp_path
    assert settings.data_dir == tmp_path / "data"
    assert settings.index_dir == tmp_path / "data" / "index"
    assert settings.faiss_dir == tmp_path / "data" / "index" / "faiss"
    assert settings.bm25_path == tmp_path / "data" / "index" / "bm25.pkl"
    assert settings.nodes_path == tmp_path / "data" / "index" / "nodes.jsonl"
    assert settings.doc_trees_path == tmp_path / "data" / "index" / "doc_trees.json"
    assert settings.log_level == "INFO"
    assert settings.llm_model == ""
    assert settings.llm_api_key == ""
    assert settings.llm_api_base == ""
    assert settings.llm_temperature == 0.2
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.chunker_type == "recursive"
    assert settings.index_mode == "flat"
    assert settings.leaf_node_type == "paragraph"
    assert settings.parent_embed_pooling == "mean"
    assert settings.retriever_k == 10
    assert settings.fusion_alpha == 0.5
    assert settings.max_tool_calls == 8
    assert settings.max_iterations == 10
    assert settings.max_context_tokens == 5000
    assert settings.keep_messages == 20
    assert settings.offline_mode is False


def test_load_settings_from_env(tmp_env_file, monkeypatch):
    """Test load_settings loading from .env file."""
    # Ensure environment is clean before test
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    base_dir = tmp_env_file.parent
    settings = load_settings(base_dir=base_dir, env_file=tmp_env_file)

    assert settings.llm_api_key == "test-key-123"
    assert settings.llm_api_base == "https://api.test.com/v1"
    assert settings.llm_model == "test-model"
    assert settings.log_level == "DEBUG"
    assert settings.chunker_params == {"chunk_size": 256, "chunk_overlap": 32}
    assert settings.retriever_k == 5


def test_is_offline_mode(monkeypatch):
    """Test is_offline_mode with various environment values."""
    # Test true values
    for value in ["1", "true", "True", "TRUE", "yes", "YES", "on", "ON"]:
        monkeypatch.setenv("OFFLINE_MODE", value)
        assert is_offline_mode() is True, f"Failed for value: {value}"

    # Test false values
    for value in ["0", "false", "False", "no", "off", "", "random"]:
        monkeypatch.setenv("OFFLINE_MODE", value)
        assert is_offline_mode() is False, f"Failed for value: {value}"

    # Test when not set
    monkeypatch.delenv("OFFLINE_MODE", raising=False)
    assert is_offline_mode() is False


def test_load_dotenv_parsing(tmp_path, monkeypatch):
    """Test load_dotenv parsing with various formats."""
    env_file = tmp_path / ".env.test"
    env_content = """
# Comment line
KEY1=value1
KEY2="quoted value"
KEY3='single quoted'
export KEY4=exported_value
KEY5=value with spaces
# Another comment
EMPTY_KEY=
    """.strip()
    env_file.write_text(env_content)

    # Clear existing env vars
    for key in ["KEY1", "KEY2", "KEY3", "KEY4", "KEY5", "EMPTY_KEY"]:
        monkeypatch.delenv(key, raising=False)

    values = load_dotenv(env_file, override=True)

    assert values["KEY1"] == "value1"
    assert values["KEY2"] == "quoted value"
    assert values["KEY3"] == "single quoted"
    assert values["KEY4"] == "exported_value"
    assert values["KEY5"] == "value with spaces"
    assert values["EMPTY_KEY"] == ""

    # Check environment variables were set
    assert os.getenv("KEY1") == "value1"
    assert os.getenv("KEY2") == "quoted value"
    assert os.getenv("KEY3") == "single quoted"
    assert os.getenv("KEY4") == "exported_value"


def test_load_dotenv_nonexistent_file(tmp_path):
    """Test load_dotenv with nonexistent file."""
    nonexistent = tmp_path / "does_not_exist.env"
    values = load_dotenv(nonexistent)
    assert values == {}


def test_configure_logging(tmp_path):
    """Test configure_logging creates log directory and configures logger."""
    from core.settings import AppSettings

    settings = AppSettings(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "data" / "index",
        faiss_dir=tmp_path / "data" / "index" / "faiss",
        bm25_path=tmp_path / "data" / "index" / "bm25.pkl",
        nodes_path=tmp_path / "data" / "index" / "nodes.jsonl",
        doc_trees_path=tmp_path / "data" / "index" / "doc_trees.json",
        log_dir=tmp_path / "logs",
        log_file=tmp_path / "logs" / "test.log",
        log_level="DEBUG",
    )

    configure_logging(settings)

    # Check log directory was created
    assert settings.log_dir.exists()

    # Check logging is configured
    import logging

    logger = logging.getLogger()
    assert logger.level == logging.DEBUG


def test_app_settings_methods(tmp_path):
    """Test AppSettings helper methods."""
    from core.settings import AppSettings

    settings = AppSettings(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "data" / "index",
        faiss_dir=tmp_path / "data" / "index" / "faiss",
        bm25_path=tmp_path / "data" / "index" / "bm25.pkl",
        nodes_path=tmp_path / "data" / "index" / "nodes.jsonl",
        doc_trees_path=tmp_path / "data" / "index" / "doc_trees.json",
        log_dir=tmp_path / "logs",
        log_file=tmp_path / "logs" / "test.log",
        log_level="INFO",
        llm_model="gpt-4",
        llm_api_key="test-key",
        llm_api_base="https://api.openai.com/v1",
        llm_temperature=0.7,
        embedding_model="text-embedding-ada-002",
        embedding_api_key="test-embed-key",
        embedding_api_base="https://api.openai.com/v1",
        chunker_type="token",
        chunker_params={"chunk_size": 512},
        retriever_k=15,
        fusion_alpha=0.6,
    )

    # Test llm_config
    llm_config = settings.llm_config()
    assert llm_config["model"] == "gpt-4"
    assert llm_config["api_key"] == "test-key"
    assert llm_config["api_base"] == "https://api.openai.com/v1"
    assert llm_config["model_config"]["temperature"] == 0.7

    # Test indexer_config
    indexer_config = settings.indexer_config()
    assert indexer_config["embedding"]["model"] == "text-embedding-ada-002"
    assert indexer_config["embedding"]["api_key"] == "test-embed-key"
    assert indexer_config["chunker"]["type"] == "token"
    assert indexer_config["chunker"]["params"] == {"chunk_size": 512}
    assert indexer_config["index_mode"] == "flat"
    assert indexer_config["leaf_node_type"] == "paragraph"
    assert indexer_config["parent_embed_pooling"] == "mean"
    assert indexer_config["retriever"]["k"] == 15
    assert indexer_config["retriever"]["alpha"] == 0.6

    # Test ensure_dirs
    settings.ensure_dirs()
    assert settings.data_dir.exists()
    assert settings.index_dir.exists()
    assert settings.faiss_dir.exists()
    assert settings.log_dir.exists()
    assert settings.bm25_path.parent.exists()
    assert settings.nodes_path.parent.exists()
    assert settings.doc_trees_path.parent.exists()
