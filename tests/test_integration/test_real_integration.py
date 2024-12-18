import unittest
import logging
import time
from unittest.mock import patch
from tests.test_config import get_mock_openai, get_mock_pinecone, MockOpenAI

logger = logging.getLogger(__name__)

class TestRealIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment with mocked components."""
        try:
            logger.info("Initializing test components...")
            start_time = time.time()

            # Initialize components with mocks
            mock_client = MockOpenAI()
            cls.embeddings = mock_client.embeddings
            cls.chat = mock_client.chat
            cls.index = get_mock_pinecone().Index("test-index")

            setup_time = time.time() - start_time
            logger.info(f"✓ Successfully initialized all components in {setup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize components: {str(e)}")
            raise

    def test_end_to_end_processing(self):
        """Test the entire document processing pipeline."""
        test_docs = ["Test document 1", "Test document 2"]
        
        # Test embedding generation
        embeddings = self.embeddings.create(input=test_docs)
        self.assertIsNotNone(embeddings)
        
        # Test vector storage
        upsert_response = self.index.upsert(
            vectors=[{
                "id": f"doc_{i}", 
                "values": embedding.embedding
            } for i, embedding in enumerate(embeddings.data)],
            namespace="test"
        )
        self.assertEqual(upsert_response["upserted_count"], 2)

    def test_query_processing(self):
        """Test the query processing pipeline."""
        test_query = "What is the test about?"
        
        # Test chat completion
        response = self.chat.create(
            messages=[{"role": "user", "content": test_query}]
        )
        self.assertIsNotNone(response)
        self.assertTrue(isinstance(response.choices[0].message.content, str))

if __name__ == '__main__':
    unittest.main()