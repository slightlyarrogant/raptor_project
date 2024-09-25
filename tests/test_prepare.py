import unittest
from unittest.mock import patch, mock_open
import numpy as np
from src.prepare.data_loader import load_text_files, preprocess_documents
from src.prepare.embeddings import embed_texts
from src.utils.config import CHUNK_SIZE_TOKENS

class TestDataLoader(unittest.TestCase):
    @patch('os.listdir')
    @patch('builtins.open', new_callable=mock_open, read_data="Test content")
    def test_load_text_files(self, mock_file, mock_listdir):
        mock_listdir.return_value = ['file1.txt', 'file2.txt', 'file3.pdf']
        docs = load_text_files('/fake/path')
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].page_content, "Test content")
        self.assertEqual(docs[0].metadata['filename'], 'file1.txt')

    def test_preprocess_documents(self):
        docs = [
            {'page_content': 'This is a test document.', 'metadata': {'filename': 'test1.txt'}},
            {'page_content': 'This is another test document.', 'metadata': {'filename': 'test2.txt'}}
        ]
        preprocessed = preprocess_documents(docs)
        self.assertIsInstance(preprocessed, list)
        self.assertTrue(all(isinstance(text, str) for text in preprocessed))

class TestEmbeddings(unittest.TestCase):
    @patch('src.prepare.embeddings.OpenAIEmbeddings')
    def test_embed_texts(self, mock_embeddings):
        mock_embeddings.return_value.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        texts = ['Test text 1', 'Test text 2']
        result = embed_texts(texts, mock_embeddings.return_value)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 3))

if __name__ == '__main__':
    unittest.main()
