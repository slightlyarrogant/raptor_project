import unittest
import os
from dotenv import load_dotenv
from src.utils.openai_client import UnifiedAIClient

# Load environment variables
load_dotenv()

@unittest.skipIf(not os.getenv("OPENAI_API_KEY"), "OpenAI API key not found")
class TestOpenAILive(unittest.TestCase):
    """Live tests using actual OpenAI API. 
    Only runs if OPENAI_API_KEY environment variable is set."""
    
    def setUp(self):
        self.client = UnifiedAIClient()

    def test_basic_functionality(self):
        """Test basic API functionality"""
        # Test embedding
        text = "This is a test document"
        embedding = self.client.get_embeddings([text])
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding[0]), 1536)

        # Test chat
        messages = [{"role": "user", "content": "Say 'test' in one word."}]
        response = self.client.chat_completion(messages)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

if __name__ == '__main__':
    unittest.main()