import unittest
from unittest.mock import patch
from src.utils.openai_client import UnifiedAIClient
from tests.test_config import MockOpenAI, TEST_EMBEDDING_DIMENSION

class TestUnifiedAIClient(unittest.TestCase):
    def setUp(self):
        # Reset singleton instance before each test
        UnifiedAIClient._instance = None
        
    def test_singleton_pattern(self):
        with patch('openai.OpenAI', return_value=MockOpenAI()):
            client1 = UnifiedAIClient()
            client2 = UnifiedAIClient()
            self.assertIs(client1, client2)

    def test_get_embeddings(self):
        with patch('openai.OpenAI', return_value=MockOpenAI()):
            client = UnifiedAIClient()
            texts = ["Test text 1", "Test text 2"]
            embeddings = client.get_embeddings(texts)
            
            self.assertEqual(len(embeddings), 2)
            self.assertEqual(len(embeddings[0]), TEST_EMBEDDING_DIMENSION)

    def test_chat_completion(self):
        with patch('openai.OpenAI', return_value=MockOpenAI()):
            client = UnifiedAIClient()
            messages = [{"role": "user", "content": "Hello"}]
            response = client.chat_completion(messages)
            
            self.assertEqual(response, "This is a mock response")

    def test_error_handling(self):
        with patch('openai.OpenAI') as mock_openai:
            mock_instance = mock_openai.return_value
            mock_instance.embeddings.create.side_effect = Exception("API Error")

            client = UnifiedAIClient()
            with self.assertRaises(Exception):
                client.get_embeddings(["Test"])

if __name__ == '__main__':
    unittest.main()