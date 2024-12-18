"""Test utilities for mocking components."""
from unittest.mock import MagicMock
from typing import Dict

def get_mock_openai():
    """Create mock OpenAI client."""
    mock = MagicMock()
    mock.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )
    mock.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Test summary"))]
    )
    return mock

def get_mock_pinecone(config: Dict = None):
    """Create mock Pinecone client."""
    mock = MagicMock()
    mock.Index.return_value = MagicMock(
        upsert=MagicMock(return_value={"upserted_count": 1}),
        query=MagicMock(return_value={"matches": []}),
        delete=MagicMock(return_value=True)
    )
    return mock 