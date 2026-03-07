"""Tests for data_processor module."""

import pytest
from langchain_core.documents import Document
from indexing.data_processor import clean_text, TextProcessor, PdfProcessor


def test_clean_text_hyphen_linebreak():
    """Test clean_text removes hyphen-linebreak patterns."""
    text = "This is an exam-\nple of text."
    cleaned = clean_text(text)
    assert cleaned == "This is an example of text."

    text2 = "multi-\nline-\nword"
    cleaned2 = clean_text(text2)
    assert cleaned2 == "multilineword"


def test_clean_text_newlines():
    """Test clean_text converts newlines to spaces."""
    text = "Hello\nWorld\nThis is a test"
    cleaned = clean_text(text)
    assert cleaned == "Hello World This is a test"


def test_clean_text_combined():
    """Test clean_text handles both hyphen-linebreaks and normal newlines."""
    text = "This is a hyphen-\nated word.\nAnd a new line."
    cleaned = clean_text(text)
    assert cleaned == "This is a hyphenated word. And a new line."


def test_clean_text_strips_whitespace():
    """Test clean_text strips leading/trailing whitespace."""
    text = "   \n  Hello World  \n  "
    cleaned = clean_text(text)
    assert cleaned == "Hello World"


def test_text_processor(tmp_path):
    """Test TextProcessor loads and processes text files."""
    text_file = tmp_path / "test.txt"
    content = "Python is amazing.\nIt has many libraries."
    text_file.write_text(content, encoding="utf-8")

    processor = TextProcessor()
    docs = processor.process(str(text_file))

    assert len(docs) > 0
    assert isinstance(docs[0], Document)

    # Verify content was processed (newlines converted to spaces)
    assert "Python is amazing." in docs[0].page_content
    assert "\n" not in docs[0].page_content  # newlines should be cleaned

    # Verify metadata
    assert "source" in docs[0].metadata


def test_text_processor_custom_encoding(tmp_path):
    """Test TextProcessor with custom encoding."""
    text_file = tmp_path / "test_utf8.txt"
    content = "Hello 世界"
    text_file.write_text(content, encoding="utf-8")

    processor = TextProcessor(encoding="utf-8")
    docs = processor.process(str(text_file))

    assert len(docs) > 0
    assert "Hello" in docs[0].page_content


def test_text_processor_nonexistent_file():
    """Test TextProcessor raises error for nonexistent file."""
    processor = TextProcessor()

    with pytest.raises(ValueError, match="TextProcessor error"):
        processor.process("/nonexistent/file.txt")


def test_pdf_processor_nonexistent_file():
    """Test PdfProcessor raises error for nonexistent file."""
    processor = PdfProcessor()

    with pytest.raises(ValueError, match="PdfProcessor error"):
        processor.process("/nonexistent/file.pdf")


def test_clean_text_empty_string():
    """Test clean_text handles empty strings."""
    assert clean_text("") == ""
    assert clean_text("   ") == ""


def test_clean_text_preserves_content():
    """Test clean_text preserves important content."""
    text = "Important content-\nwith multiple\nlines and data."
    cleaned = clean_text(text)

    assert "Important" in cleaned
    assert "content" in cleaned
    assert "multiple" in cleaned
    assert "lines" in cleaned
    assert "data" in cleaned
