"""Document processing module for loading and splitting documents"""

from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Handles document loading and processing"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load document(s) from a URL"""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory (one Document per page)"""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a TXT file"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a single PDF file (one Document per page)"""
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF directories, PDF files, or TXT files

        Args:
            sources: List of URLs, PDF folder paths, PDF file paths, or TXT file paths

        Returns:
            List of loaded documents

        Example:
            >>> processor.load_documents(["https://ejemplo.com", "data", "notas.txt"])
        """
        docs: List[Document] = []
        for src in sources:
            if src.startswith(("http://", "https://")):
                docs.extend(self.load_from_url(src))
                continue

            path = Path(src)
            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower() == ".pdf":
                docs.extend(self.load_from_pdf(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))
            else:
                raise ValueError(
                    f"Unsupported source type: {src}. "
                    "Use URL, .pdf file, .txt file, or PDF directory."
                )
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks

        Args:
            documents: List of documents to split

        Returns:
            List of split documents
        """
        return self.splitter.split_documents(documents)

    def process_urls(self, sources: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents

        Args:
            sources: List of URLs, PDF folders, PDF files, or TXT files

        Returns:
            List of processed document chunks
        """
        docs = self.load_documents(sources)
        return self.split_documents(docs)