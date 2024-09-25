from typing import List
import numpy as np
from langchain_openai import OpenAIEmbeddings
from src.utils.config import EMBEDDING_MODEL, VERBOSE

def create_embedding_model() -> OpenAIEmbeddings:
    """Create and return an OpenAIEmbeddings model."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)

def embed_texts(texts: List[str], embd: OpenAIEmbeddings) -> np.ndarray:
    """
    Generate embeddings for a list of text documents.

    Parameters:
    - texts: List[str], a list of text documents to be embedded.
    - embd: The embedding model to use.

    Returns:
    - numpy.ndarray: An array of embeddings for the given text documents.
    """
    if VERBOSE:
        print(f"Generating embeddings for {len(texts)} texts")
    text_embeddings = embd.embed_documents(texts)
    text_embeddings_np = np.array(text_embeddings)
    return text_embeddings_np
