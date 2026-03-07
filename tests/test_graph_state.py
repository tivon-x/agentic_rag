"""Tests for graph_state module."""

from agent.graph_state import accumulate_or_reset, set_union


def test_accumulate_or_reset_accumulate():
    """Test accumulate_or_reset accumulates normally without reset flag."""
    existing = [{"answer": "first"}, {"answer": "second"}]
    new = [{"answer": "third"}]

    result = accumulate_or_reset(existing, new)

    assert len(result) == 3
    assert result[0] == {"answer": "first"}
    assert result[1] == {"answer": "second"}
    assert result[2] == {"answer": "third"}


def test_accumulate_or_reset_reset():
    """Test accumulate_or_reset resets when __reset__ flag is present."""
    existing = [{"answer": "first"}, {"answer": "second"}]
    new = [{"__reset__": True}]

    result = accumulate_or_reset(existing, new)

    assert len(result) == 0


def test_accumulate_or_reset_reset_with_data():
    """Test accumulate_or_reset resets even when reset item has other data."""
    existing = [{"answer": "first"}, {"answer": "second"}]
    new = [{"__reset__": True, "answer": "new"}]

    result = accumulate_or_reset(existing, new)

    assert len(result) == 0


def test_accumulate_or_reset_multiple_new_items():
    """Test accumulate_or_reset with multiple new items."""
    existing = [{"answer": "first"}]
    new = [{"answer": "second"}, {"answer": "third"}]

    result = accumulate_or_reset(existing, new)

    assert len(result) == 3
    assert result == [
        {"answer": "first"},
        {"answer": "second"},
        {"answer": "third"},
    ]


def test_accumulate_or_reset_empty_existing():
    """Test accumulate_or_reset with empty existing list."""
    existing = []
    new = [{"answer": "first"}]

    result = accumulate_or_reset(existing, new)

    assert len(result) == 1
    assert result[0] == {"answer": "first"}


def test_accumulate_or_reset_empty_new():
    """Test accumulate_or_reset with empty new list."""
    existing = [{"answer": "first"}]
    new = []

    result = accumulate_or_reset(existing, new)

    assert len(result) == 1
    assert result[0] == {"answer": "first"}


def test_accumulate_or_reset_reset_false():
    """Test accumulate_or_reset when __reset__ is explicitly False."""
    existing = [{"answer": "first"}]
    new = [{"__reset__": False, "answer": "second"}]

    result = accumulate_or_reset(existing, new)

    # Should accumulate since __reset__ is False (falsy)
    assert len(result) == 2


def test_set_union():
    """Test set_union combines two sets."""
    set_a = {"apple", "banana"}
    set_b = {"cherry", "date"}

    result = set_union(set_a, set_b)

    assert result == {"apple", "banana", "cherry", "date"}


def test_set_union_overlapping():
    """Test set_union with overlapping sets."""
    set_a = {"apple", "banana", "cherry"}
    set_b = {"banana", "cherry", "date"}

    result = set_union(set_a, set_b)

    assert result == {"apple", "banana", "cherry", "date"}


def test_set_union_empty_sets():
    """Test set_union with empty sets."""
    assert set_union(set(), set()) == set()
    assert set_union({"apple"}, set()) == {"apple"}
    assert set_union(set(), {"banana"}) == {"banana"}


def test_set_union_identical_sets():
    """Test set_union with identical sets."""
    set_a = {"apple", "banana"}
    set_b = {"apple", "banana"}

    result = set_union(set_a, set_b)

    assert result == {"apple", "banana"}


def test_set_union_preserves_elements():
    """Test set_union preserves all unique elements."""
    set_a = {"a", "b", "c"}
    set_b = {"d", "e", "f"}

    result = set_union(set_a, set_b)

    for item in set_a:
        assert item in result
    for item in set_b:
        assert item in result
