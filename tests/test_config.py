"""Test configuration with mock values and fixtures"""
import os
from unittest.mock import MagicMock
from dataclasses import dataclass

@dataclass
class MockEmbedding:
    embedding: list

@dataclass
class MockMessage:
    content: str

@dataclass
class MockChoice:
    message: MockMessage

# Test constants
TEST_EMBEDDING_DIMENSION = 1536
TEST_CHUNK_SIZE = 1000

# Store original API key if it exists
ORIGINAL_OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Test environment variables - only set for mock tests
TEST_ENV = {
    "PINECONE_API_KEY": "test-pinecone-key",
    "PINECONE_INDEX_NAME": "test-index",
    "PINECONE_DIMENSION": "1536",
    "PINECONE_METRIC": "cosine",
    "PINECONE_CLOUD": "test-cloud",
    "PINECONE_REGION": "test-region",
    "PINECONE_ENVIRONMENT": "test-env"
}

def setup_test_env():
    """Set up test environment variables for mock tests"""
    # Save original OpenAI key
    global ORIGINAL_OPENAI_KEY
    ORIGINAL_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    
    # Set test environment variables
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    for key, value in TEST_ENV.items():
        os.environ[key] = value

def restore_env():
    """Restore original environment variables"""
    if ORIGINAL_OPENAI_KEY:
        os.environ["OPENAI_API_KEY"] = ORIGINAL_OPENAI_KEY

class MockOpenAI:
    def __init__(self):
        self.embeddings = self.Embeddings()
        self.chat = self.Chat()

    class Embeddings:
        def create(self, input, model=None):
            return MagicMock(
                data=[
                    MockEmbedding(embedding=[0.1] * TEST_EMBEDDING_DIMENSION)
                    for _ in input
                ]
            )

    class Chat:
        def create(self, messages, model=None, temperature=0.7):
            return MagicMock(
                choices=[
                    MockChoice(message=MockMessage(content="This is a mock response"))
                ]
            )

class MockPineconeIndex:
    """Mock Pinecone Index with all required methods"""
    def __init__(self, name="test-index"):
        self.name = name
        self.dimension = 1536
        self.metric = "cosine"
        self.environment = "test-env"
        self._vectors = {}  # Store vectors in memory for testing

    def upsert(self, vectors, namespace=""):
        """Mock vector upsert"""
        for vector in vectors:
            self._vectors[vector["id"]] = {
                "values": vector["values"],
                "metadata": vector.get("metadata", {})
            }
        return {"upserted_count": len(vectors)}

    def query(self, vector=None, queries=None, top_k=10, namespace="", include_values=False):
        """Mock vector query"""
        return {
            "matches": [
                {
                    "id": f"test_doc_{i}",
                    "score": 0.9 - (i * 0.1),
                    "values": [0.1] * self.dimension if include_values else None,
                    "metadata": {"text": f"Test document {i}"}
                } for i in range(min(top_k, len(self._vectors) or 3))
            ]
        }

    def describe_index_stats(self):
        """Mock index statistics"""
        return {
            "dimension": self.dimension,
            "index_fullness": 0.0,
            "total_vector_count": len(self._vectors),
            "namespaces": {
                "": {"vector_count": len(self._vectors)}
            }
        }

    def delete(self, ids=None, namespace="", deleteAll=False):
        """Mock vector deletion"""
        if deleteAll:
            self._vectors.clear()
        elif ids:
            for id in ids:
                self._vectors.pop(id, None)
        return {}

class MockPinecone:
    """Mock Pinecone client with proper initialization"""
    def __init__(self, api_key=None):
        self.api_key = api_key or "test-api-key"
        self._indexes = {}

    def Index(self, name, host=None):
        """Create or get a mock index"""
        if name not in self._indexes:
            self._indexes[name] = MockPineconeIndex(name)
        return self._indexes[name]

class MockPineconeManager:
    """Mock PineconeManager for testing"""
    def __init__(self, config=None):
        self.index = MockPineconeIndex()
        self.namespace = config.get('namespace', 'test-namespace')
        self.dimension = config.get('dimension', 1536)
        self.config = config or {}
        
    def initialize(self):
        """Mock initialization"""
        return True

    def store_vectors(self, vectors, metadata=None):
        """Mock vector storage"""
        return self.index.upsert(vectors=vectors, namespace=self.namespace)

    def query_vectors(self, query_vector, top_k=5):
        """Mock vector query"""
        return self.index.query(vector=query_vector, top_k=top_k, namespace=self.namespace)

    def delete_vectors(self, ids=None):
        """Mock vector deletion"""
        return self.index.delete(ids=ids, namespace=self.namespace)

# Create default test config for mocks
DEFAULT_TEST_CONFIG = {
    'pinecone_index': 'test-index',
    'namespace': 'test-namespace',
    'environment': 'test-env',
    'dimension': TEST_EMBEDDING_DIMENSION,
    'metric': 'cosine',
    'api_key': 'test-key',
    'test_mode': True
}

def get_mock_pinecone(config=None):
    """Get a configured mock Pinecone instance with manager"""
    config = config or DEFAULT_TEST_CONFIG  # Use default config if none provided
    mock_pinecone = MockPinecone()
    mock_pinecone.manager = MockPineconeManager(config)
    # Set the store attribute that PineconeManager expects
    mock_pinecone.store = mock_pinecone.Index(config.get('pinecone_index', 'test-index'))
    return mock_pinecone

def get_mock_openai():
    return MockOpenAI()

# Create instances for direct use in tests with default config
mock_openai_instance = MockOpenAI()
mock_pinecone_instance = get_mock_pinecone(DEFAULT_TEST_CONFIG)

# Update test environment configuration
TEST_ENV.update({
    "PINECONE_API_KEY": "test-pinecone-key",
    "PINECONE_INDEX_NAME": "test-index",
    "PINECONE_ENVIRONMENT": "test-env",
    "PINECONE_DIMENSION": "1536",
    "PINECONE_METRIC": "cosine"
})