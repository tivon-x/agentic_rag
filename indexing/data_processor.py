"""
数据处理模块，包含不同类型文件的处理器
每个处理器都继承自DataProcessor基类，并实现process方法来处理特定类型的文件。

当前实现了PdfProcessor来处理PDF文件，使用PyPDFLoader加载PDF内容，并进行文本清洗以去除换行符和连字符换行。
"""

from abc import ABC, abstractmethod
from langchain_core.documents import Document
import re
from langchain_community.document_loaders import (
    PyPDFLoader, 
)

class DataProcessor(ABC):
    """Base class for all data processors"""
    @abstractmethod
    def process(self, file_path: str) -> list[Document]:
        """
        处理文件并返回文档列表
        Args:
            file_path (str): 文件路径
        Returns:
            List[Document]: 处理后的文档列表
        """
        pass


class PdfProcessor(DataProcessor):
    def process(self, file_path: str) -> list[Document]:
        try:
            loader = PyPDFLoader(file_path)
            pages = []
            for page in loader.lazy_load():
                page.page_content = clean_text(page.page_content)
                pages.append(page)
            return pages
        except Exception as e:
            raise ValueError(f"PdfProcessor error: {e}")
        


def clean_text(text: str) -> str:
    """
    文本清洗函数：
    1. 合并被换行断开的单词（如 xxx-\nxxx → xxxxxx）
    2. 将换行符转换为空格
    """
    # 第一步：处理连字符换行
    text = re.sub(r'-\n', '', text)
    
    # 第二步：处理普通换行
    text = re.sub(r'\n', ' ', text)
    
    return text.strip()