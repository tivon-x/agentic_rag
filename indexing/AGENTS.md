# INDEXING MODULE (indexing/)

## OVERVIEW
Document ingestion + chunking + embeddings + vector/BM25 retrieval pipeline.

## STRUCTURE
```
indexing/
├── indexer.py        # orchestration of ingestion → chunk → store
├── data_processor.py # PDF loader + text cleanup
├── chunker.py        # recursive/token/semantic chunkers
├── embeddings.py     # OpenAI-compatible embeddings
├── vectorstore.py    # FAISS-backed store
├── retriever.py      # similarity + BM25 + fusion retrievers
└── bm25_index.py     # BM25 index creation
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| End-to-end indexing | indexer.py | main pipeline class |
| PDF loading | data_processor.py | PyPDFLoader + cleanup |
| Chunk strategies | chunker.py | recursive/token/semantic |
| Embeddings | embeddings.py | requires API key/base URL |
| Vector store | vectorstore.py | FAISS local store |
| Retrieval | retriever.py | BM25 + fusion |

## CONVENTIONS
- Embeddings require API key/base URL; no local HF fallback is implemented.
- Chunker selection is via `mappers.CHUNER_MAPPING`.

## ANTI-PATTERNS
- Do not bypass `VectorStore` directly; keep store access centralized.
