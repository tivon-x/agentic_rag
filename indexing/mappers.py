# Indexer Mapper (file extensions and chunker types)
from indexing.chunker import RecursiveChunker, SemanticNLTKChunker, TokenChunker
from indexing.data_processor import PdfProcessor, TextProcessor


LOADER_MAPPING = {
    ".pdf": (PdfProcessor, {}),
    ".md": (TextProcessor, {}),
    ".txt": (TextProcessor, {}),
}

CHUNKER_MAPPING = {
    "recursive": (RecursiveChunker, {}),
    "token": (TokenChunker, {}),
    "SemanticNLTKChunker": (SemanticNLTKChunker, {}),
}
