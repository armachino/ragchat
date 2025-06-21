import os
from typing import List

from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
import bs4

def get_loader_from_dir(folder_path: str = "./docs") -> List[Document]:
    """
    Loads all PDF documents from a given folder.

    Args:
        folder_path (str): Path to the folder containing PDF files.

    Returns:
        List[Document]: List of LangChain Document objects.
    """
    loader = DirectoryLoader(
        path=folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader, # type: ignore
        recursive=True
    )
    docs = loader.load()
    return docs

def get_loader_from_url(url: str = "https://lilianweng.github.io/posts/2023-06-23-agent/") -> List[Document]:
    """
    Loads all PDF documents from a given folder.

    Args:
        folder_path (str): Path to the folder containing PDF files.

    Returns:
        List[Document]: List of LangChain Document objects.
    """
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer( # type: ignore
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    return docs 

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(base_dir)
    folder_path = os.path.join(base_dir, "..","..", "docs")  # adjust path to go up
    print(folder_path)
    print(get_loader_from_dir(folder_path))
