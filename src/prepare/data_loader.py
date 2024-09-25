import os
from typing import List
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from src.utils.config import CHUNK_SIZE_TOKENS, VERBOSE

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def load_text_files(directory: str) -> List[Document]:
    """Loads all text files from the specified directory."""
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                docs.append(Document(page_content=content, metadata={"filename": filename}))
    return docs

def preprocess_documents(docs: List[Document]) -> List[str]:
    """Concatenate and split documents into chunks."""
    if VERBOSE:
        print("Concatenating and splitting documents")
    
    d_sorted = sorted(docs, key=lambda x: x.metadata["filename"])
    d_reversed = list(reversed(d_sorted))
    concatenated_content = "\n\n\n --- \n\n\n".join([doc.page_content for doc in d_reversed])
    
    if VERBOSE:
        print(f"Num tokens in all context: {num_tokens_from_string(concatenated_content)}")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE_TOKENS, chunk_overlap=0
    )
    texts_split = text_splitter.split_text(concatenated_content)
    
    return texts_split

def load_and_preprocess_documents(directory: str) -> List[str]:
    """Load documents from a directory and preprocess them."""
    if VERBOSE:
        print(f"Loading and processing documents from {directory}")
    
    docs = load_text_files(directory)
    preprocessed_texts = preprocess_documents(docs)
    
    return preprocessed_texts
