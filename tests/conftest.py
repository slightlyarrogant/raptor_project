import pytest
from tests.test_config import get_mock_pinecone, get_mock_openai, DEFAULT_TEST_CONFIG

@pytest.fixture
def mock_pinecone():
    return get_mock_pinecone(DEFAULT_TEST_CONFIG)

@pytest.fixture
def mock_openai():
    return get_mock_openai() 