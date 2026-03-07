"""Tests for persistence module."""

import pytest
from indexing.bm25_index import create_bm25_bundle
from core.persistence import save_bm25_bundle, load_bm25_bundle


def test_save_load_bm25_bundle_roundtrip(tmp_path, sample_documents):
    """Test round-trip save and load of BM25Bundle."""
    # Create a BM25 bundle
    bundle = create_bm25_bundle(sample_documents)

    # Save to temporary path
    save_path = tmp_path / "test_bm25.pkl"
    save_bm25_bundle(save_path, bundle)

    # Verify file was created
    assert save_path.exists()

    # Load the bundle back
    loaded_bundle = load_bm25_bundle(save_path)

    # Verify the loaded bundle matches original
    assert len(loaded_bundle.documents) == len(bundle.documents)
    assert len(loaded_bundle.tokenized_corpus) == len(bundle.tokenized_corpus)

    # Verify document contents
    for orig_doc, loaded_doc in zip(
        bundle.documents, loaded_bundle.documents, strict=False
    ):
        assert orig_doc.page_content == loaded_doc.page_content
        assert orig_doc.metadata == loaded_doc.metadata

    # Verify loaded bundle can still query
    results = loaded_bundle.query("Python", k=2)
    assert len(results) > 0


def test_save_bm25_bundle_creates_directory(tmp_path, sample_documents):
    """Test save_bm25_bundle creates parent directory if needed."""
    bundle = create_bm25_bundle(sample_documents)

    # Use a nested path that doesn't exist yet
    nested_path = tmp_path / "nested" / "dir" / "bm25.pkl"

    save_bm25_bundle(nested_path, bundle)

    # Verify file and directories were created
    assert nested_path.exists()
    assert nested_path.parent.exists()


def test_save_bm25_bundle_atomic_write(tmp_path, sample_documents):
    """Test save_bm25_bundle uses atomic write (no partial files)."""
    bundle = create_bm25_bundle(sample_documents)
    save_path = tmp_path / "atomic_test.pkl"

    # Save the bundle
    save_bm25_bundle(save_path, bundle)

    # Verify no .tmp file exists after save
    tmp_file = save_path.with_suffix(save_path.suffix + ".tmp")
    assert not tmp_file.exists()

    # Verify the main file exists
    assert save_path.exists()


def test_load_bm25_bundle_rebuilds_index(tmp_path, sample_documents):
    """Test load_bm25_bundle rebuilds the BM25 index."""
    bundle = create_bm25_bundle(sample_documents)
    save_path = tmp_path / "rebuild_test.pkl"

    # Save bundle
    save_bm25_bundle(save_path, bundle)

    # Load bundle
    loaded_bundle = load_bm25_bundle(save_path)

    # Verify index was rebuilt (not None)
    assert loaded_bundle._bm25 is not None

    # Verify index works
    scores = loaded_bundle.bm25_index.get_scores("Python".split())
    assert len(scores) == len(loaded_bundle.documents)


def test_load_bm25_bundle_nonexistent_file(tmp_path):
    """Test load_bm25_bundle raises error for nonexistent file."""
    nonexistent = tmp_path / "does_not_exist.pkl"

    with pytest.raises(FileNotFoundError):
        load_bm25_bundle(nonexistent)


def test_save_load_empty_bundle(tmp_path):
    """Test save and load with empty BM25 bundle."""
    empty_bundle = create_bm25_bundle([])
    save_path = tmp_path / "empty.pkl"

    save_bm25_bundle(save_path, empty_bundle)
    loaded_bundle = load_bm25_bundle(save_path)

    assert len(loaded_bundle.documents) == 0
    assert len(loaded_bundle.tokenized_corpus) == 0


def test_save_overwrite_existing(tmp_path, sample_documents):
    """Test save_bm25_bundle overwrites existing file."""
    bundle1 = create_bm25_bundle(sample_documents[:2])
    bundle2 = create_bm25_bundle(sample_documents[2:])

    save_path = tmp_path / "overwrite.pkl"

    # Save first bundle
    save_bm25_bundle(save_path, bundle1)
    loaded1 = load_bm25_bundle(save_path)
    assert len(loaded1.documents) == 2

    # Save second bundle (overwrite)
    save_bm25_bundle(save_path, bundle2)
    loaded2 = load_bm25_bundle(save_path)
    assert len(loaded2.documents) == 3

    # Verify content is from second bundle
    assert loaded2.documents[0].page_content == sample_documents[2].page_content
