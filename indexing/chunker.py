"""
Text chunking module providing multiple chunking strategies for different document types and use cases.
- RecursiveChunker: Recursive character-based text chunker, suitable for general text, preserves metadata.
- TokenChunker: Recursive text chunker based on token encoding, suitable for scenarios requiring precise token count control.
- SemanticNLTKChunker: Smart semantic chunker based on NLTK, supports mixed Chinese-English text with intelligent sentence segmentation.
Each chunker implements the abstract Chunker base class's chunk method, accepting a Document list as input and returning a new Document list as output. Users can select the appropriate chunker based on their needs.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from abc import ABC, abstractmethod
from langchain_core.documents import Document
import nltk
from nltk.tokenize import sent_tokenize
import jieba


class Chunker(ABC):
    @abstractmethod
    def chunk(self, docs: list[Document]) -> list[Document]:
        """
        Chunk documents
        Args:
            docs (List[Document]): Input document list
        Returns:
            list[Document]: Chunked document list
        """
        pass


class RecursiveChunker(Chunker):
    """Recursive character-based text chunker"""

    def __init__(self, chunk_size=512, chunk_overlap=64):
        """
        Initialize chunker
        Args:
            chunk_size (int): Maximum characters per chunk
            chunk_overlap (int): Overlap characters between chunks
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n## ", "\n# ", "\n\n", "\n", "。", "!", "?", " ", ""],
        )  # Note: metadata is preserved

    def chunk(self, docs: list[Document]) -> list[Document]:
        return self.splitter.split_documents(docs)


class TokenChunker(Chunker):
    """Recursive text chunker based on token encoding"""

    def __init__(self, chunk_size=512, chunk_overlap=64):
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self, docs: list[Document]) -> list[Document]:
        return self.splitter.split_documents(docs)


class SemanticNLTKChunker(Chunker):
    """Smart semantic chunker based on NLTK, supports mixed Chinese-English text"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        language: str = "chinese",
        use_jieba: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        self.use_jieba = use_jieba

        # Initialize Chinese word segmenter
        if self.language == "chinese" and self.use_jieba:
            jieba.initialize()

    def _chinese_sentence_split(self, text: str) -> list[str]:
        """Smart sentence segmentation based on jieba"""
        if not self.use_jieba:
            return [text]

        delimiters = {"。", "！", "？", "；", "…"}
        sentences = []
        buffer = []

        for word in jieba.cut(text):
            buffer.append(word)
            if word in delimiters:
                sentences.append("".join(buffer))
                buffer = []

        if buffer:  # Handle sentences without ending punctuation
            sentences.append("".join(buffer))
        return sentences

    def split_text(self, doc: Document) -> list[Document]:
        """Multi-language sentence segmentation logic"""
        sentences = []
        if self.language == "chinese":
            sentences = self._chinese_sentence_split(doc.page_content)
        else:
            nltk.download("punkt_tab")
            sentences = sent_tokenize(doc.page_content, language=self.language)

        """Dynamically merge sentences and preserve character overlap"""
        chunks = []
        current_chunk = []
        current_length = 0
        overlap_buffer = []

        for sent in sentences:
            sent_len = len(
                sent.split(" ")
            )  # Calculate length based on space-separated tokens

            # Trigger chunking condition
            if current_length + sent_len > self.chunk_size:
                if current_chunk:
                    chunks.append("".join(current_chunk))

                    # Calculate overlap
                    overlap_buffer = []
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) > self.chunk_overlap:
                            break
                        overlap_buffer.append(s)
                        overlap_length += len(s)

                    current_chunk = list(reversed(overlap_buffer))
                    current_length = overlap_length

            current_chunk.append(sent)
            current_length += sent_len

        # Handle remaining content
        if current_chunk:
            chunks.append("".join(current_chunk))
        return [
            Document(page_content=chunk, metadata=doc.metadata.copy())
            for chunk in chunks
        ]

    def chunk(self, docs: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for doc in docs:
            chunks.extend(self.split_text(doc))
        return chunks
