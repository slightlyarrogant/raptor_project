import unittest
from unittest.mock import patch, mock_open
import numpy as np
from src.prepare.data_loader import load_text_files, preprocess_documents
from src.prepare.embeddings import embed_texts
from src.utils.config import CHUNK_SIZE_TOKENS
from dataclasses import dataclass

@dataclass
class Document:
    page_content: str
    metadata: dict

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.test_docs = [
            Document(page_content='Test document 1', metadata={'filename': 'test1.txt'}),
            Document(page_content='Test document 2', metadata={'filename': 'test2.txt'})
        ]

    @patch('os.listdir')
    @patch('builtins.open', new_callable=mock_open, read_data="Test content")
    def test_load_text_files(self, mock_file, mock_listdir):
        mock_listdir.return_value = ['file1.txt', 'file2.txt', 'file3.pdf']
        docs = load_text_files('/fake/path')
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].page_content, "Test content")
        self.assertEqual(docs[0].metadata['filename'], 'file1.txt')

    def test_preprocess_documents(self):
        preprocessed = preprocess_documents(self.test_docs)
        self.assertIsInstance(preprocessed, list)
        self.assertTrue(all(isinstance(text, str) for text in preprocessed))
        
    def test_preprocess_special_chars(self):
        docs_with_special = [
            Document(page_content='Test @#$ content', metadata={'filename': 'test1.txt'}),
            Document(page_content='Multiple\nline\ntext', metadata={'filename': 'test2.txt'})
        ]
        preprocessed = preprocess_documents(docs_with_special)
        self.assertTrue(all(isinstance(text, str) for text in preprocessed))
        self.assertTrue(all(len(text) > 0 for text in preprocessed))

class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        # Reset OpenAI client before each test
        from src.utils.openai_client import OpenAIClient
        OpenAIClient._instance = None

    @patch('src.utils.openai_client.OpenAIClient')
    def test_embed_texts(self, mock_client):
        mock_instance = mock_client.return_value
        mock_instance.get_embeddings.return_value = [
            [0.1] * 1536,
            [0.2] * 1536
        ]
        
        texts = ['Test text 1', 'Test text 2']
        result = embed_texts(texts, client=mock_instance)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 1536))

    def test_embed_texts_error_handling(self):
        """Test error handling for invalid inputs."""
        with self.assertRaises(ValueError):
            embed_texts([])  # Test empty list
        with self.assertRaises((ValueError, TypeError)):  # Accept either error type
            embed_texts(None)  # Test None input

if __name__ == '__main__':
    unittest.main()
