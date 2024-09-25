import unittest
from unittest.mock import patch, MagicMock
from src.inference.query import similarity_search, format_search_results
from src.inference.rag_chain import create_rag_chain, query_rag_chain

class TestQuery(unittest.TestCase):
    @patch('src.inference.query.embed_texts')
    @patch('src.inference.query.query_pinecone')
    def test_similarity_search(self, mock_query_pinecone, mock_embed_texts):
        mock_embed_texts.return_value = [[0.1, 0.2, 0.3]]
        mock_query_pinecone.return_value = {
            'matches': [
                {'metadata': {'text': 'Test text', 'level': 1}, 'score': 0.9}
            ]
        }
        
        results = similarity_search('test query', MagicMock(), MagicMock())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['text'], 'Test text')
        self.assertEqual(results[0]['score'], 0.9)

    def test_format_search_results(self):
        results = [
            {'text': 'Test text 1', 'score': 0.9, 'metadata': {'level': 1}},
            {'text': 'Test text 2', 'score': 0.8, 'metadata': {'level': 2}}
        ]
        formatted = format_search_results(results)
        self.assertIsInstance(formatted, str)
        self.assertIn('Test text 1', formatted)
        self.assertIn('Test text 2', formatted)

class TestRAGChain(unittest.TestCase):
    @patch('src.inference.rag_chain.similarity_search')
    @patch('langchain.hub.pull')
    def test_create_rag_chain(self, mock_hub_pull, mock_similarity_search):
        mock_hub_pull.return_value = MagicMock()
        mock_similarity_search.return_value = [
            {'text': 'Test context', 'score': 0.9, 'metadata': {'level': 1}}
        ]
        
        rag_chain = create_rag_chain(MagicMock(), MagicMock(), MagicMock())
        self.assertIsNotNone(rag_chain)

    @patch('src.inference.rag_chain.create_rag_chain')
    def test_query_rag_chain(self, mock_create_rag_chain):
        mock_rag_chain = MagicMock()
        mock_rag_chain.invoke.return_value = "Test answer"
        mock_create_rag_chain.return_value = mock_rag_chain
        
        answer = query_rag_chain(mock_rag_chain, "Test question")
        self.assertEqual(answer, "Test answer")

if __name__ == '__main__':
    unittest.main()
