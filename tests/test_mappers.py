"""Tests for mappers module."""

from __future__ import annotations

from indexing.chunker import RecursiveChunker, SemanticNLTKChunker, TokenChunker
from indexing.data_processor import PdfProcessor, TextProcessor
from indexing.mappers import CHUNKER_MAPPING, LOADER_MAPPING


def test_loader_mapping_has_pdf():
    """Test LOADER_MAPPING contains .pdf extension."""
    assert ".pdf" in LOADER_MAPPING
    processor_class, params = LOADER_MAPPING[".pdf"]
    assert processor_class is PdfProcessor
    assert isinstance(params, dict)


def test_loader_mapping_has_markdown():
    """Test LOADER_MAPPING contains .md extension."""
    assert ".md" in LOADER_MAPPING
    processor_class, params = LOADER_MAPPING[".md"]
    assert processor_class is TextProcessor
    assert isinstance(params, dict)


def test_loader_mapping_has_txt():
    """Test LOADER_MAPPING contains .txt extension."""
    assert ".txt" in LOADER_MAPPING
    processor_class, params = LOADER_MAPPING[".txt"]
    assert processor_class is TextProcessor
    assert isinstance(params, dict)


def test_loader_mapping_keys():
    """Test LOADER_MAPPING has all required file extension keys."""
    required_keys = {".pdf", ".md", ".txt"}
    assert required_keys.issubset(LOADER_MAPPING.keys())


def test_chunker_mapping_has_recursive():
    """Test CHUNKER_MAPPING contains recursive chunker."""
    assert "recursive" in CHUNKER_MAPPING
    chunker_class, params = CHUNKER_MAPPING["recursive"]
    assert chunker_class is RecursiveChunker
    assert isinstance(params, dict)


def test_chunker_mapping_has_token():
    """Test CHUNKER_MAPPING contains token chunker."""
    assert "token" in CHUNKER_MAPPING
    chunker_class, params = CHUNKER_MAPPING["token"]
    assert chunker_class is TokenChunker
    assert isinstance(params, dict)


def test_chunker_mapping_has_semantic():
    """Test CHUNKER_MAPPING contains SemanticNLTKChunker."""
    assert "SemanticNLTKChunker" in CHUNKER_MAPPING
    chunker_class, params = CHUNKER_MAPPING["SemanticNLTKChunker"]
    assert chunker_class is SemanticNLTKChunker
    assert isinstance(params, dict)


def test_chunker_mapping_keys():
    """Test CHUNKER_MAPPING has all required chunker type keys."""
    required_keys = {"recursive", "token", "SemanticNLTKChunker"}
    assert required_keys.issubset(CHUNKER_MAPPING.keys())


def test_loader_mapping_tuple_format():
    """Test LOADER_MAPPING entries are tuples of (class, dict)."""
    for ext, (processor_class, params) in LOADER_MAPPING.items():
        assert callable(processor_class), f"Processor class for {ext} is not callable"
        assert isinstance(params, dict), f"Params for {ext} is not a dict"


def test_chunker_mapping_tuple_format():
    """Test CHUNKER_MAPPING entries are tuples of (class, dict)."""
    for chunker_type, (chunker_class, params) in CHUNKER_MAPPING.items():
        assert callable(chunker_class), (
            f"Chunker class for {chunker_type} is not callable"
        )
        assert isinstance(params, dict), f"Params for {chunker_type} is not a dict"


def test_loader_mapping_processor_classes():
    """Test LOADER_MAPPING contains valid processor classes."""
    # Verify PDF processor
    pdf_class, _ = LOADER_MAPPING[".pdf"]
    assert pdf_class.__name__ == "PdfProcessor"

    # Verify text processors
    md_class, _ = LOADER_MAPPING[".md"]
    txt_class, _ = LOADER_MAPPING[".txt"]
    assert md_class.__name__ == "TextProcessor"
    assert txt_class.__name__ == "TextProcessor"


def test_chunker_mapping_chunker_classes():
    """Test CHUNKER_MAPPING contains valid chunker classes."""
    recursive_class, _ = CHUNKER_MAPPING["recursive"]
    token_class, _ = CHUNKER_MAPPING["token"]
    semantic_class, _ = CHUNKER_MAPPING["SemanticNLTKChunker"]

    assert recursive_class.__name__ == "RecursiveChunker"
    assert token_class.__name__ == "TokenChunker"
    assert semantic_class.__name__ == "SemanticNLTKChunker"
