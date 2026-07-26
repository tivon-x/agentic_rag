import pytest
from langchain_core.documents import Document

from indexing.bm25_index import (
    create_bm25_bundle,
    create_lexical_store,
    tokenize_mixed,
)


def test_mixed_tokenizer_keeps_acronyms_numbers_and_segments_chinese():
    tokens = tokenize_mixed("RRF-60 跨章节检索 Recall@10 提升 8%")

    assert "rrf-60" in tokens
    assert "10" in tokens
    assert "8%" in tokens
    assert "检索" in tokens
    assert "章节" in tokens


def test_mixed_bm25_retrieves_chinese_term_without_spaces():
    documents = [
        Document(page_content="残差学习通过恒等映射缓解深层网络退化。"),
        Document(page_content="循环神经网络处理序列隐藏状态。"),
    ]
    bundle = create_bm25_bundle(documents, tokenizer="mixed_v1")

    results = bundle.query("残差学习恒等映射", k=1)

    assert results[0] == documents[0]


def test_lexical_store_rejects_query_tokenizer_mismatch():
    bundle = create_bm25_bundle(
        [Document(page_content="metadata prefixed retrieval")],
        tokenizer="mixed_v1",
    )

    with pytest.raises(ValueError, match="rebuild the index"):
        create_lexical_store(
            "bm25",
            bundle=bundle,
            tokenizer="whitespace_v1",
        )
