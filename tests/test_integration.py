import unittest
import logging
from src.tree.tree_manager import TreeManager
from src.utils.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class TestIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test environment with real components."""
        self.config = DEFAULT_CONFIG.copy()
        self.tree_manager = TreeManager(self.config)

    def test_tree_creation(self):
        """Test tree creation from documents."""
        documents = [
            {'id': '1', 'text': 'Test document 1', 'name': 'test1.txt'},
            {'id': '2', 'text': 'Test document 2', 'name': 'test2.txt'}
        ]
        metadata = [
            {
                "source": "test1",
                "file_name": "test1.txt",
                "file_index": 0,
                "chunk_index": 0,
                "total_chunks": 2
            },
            {
                "source": "test2",
                "file_name": "test2.txt",
                "file_index": 1,
                "chunk_index": 1,
                "total_chunks": 2
            }
        ]

        logger.info("Starting tree creation test...")
        result = self.tree_manager.process_documents(documents, metadata)
        logger.info("Tree creation result: %s", result)

        self.assertIn('tree', result)
        self.assertIn('nodes', result['tree'])
        self.assertGreater(len(result['tree']['nodes']), 0, "No nodes created in the tree")

    def test_document_metadata(self):
        """Test document metadata handling."""
        documents = [
            {'id': '1', 'text': 'Test document 1', 'name': 'test1.txt'},
        ]
        metadata = [
            {
                "source": "test1",
                "file_name": "test1.txt",
                "file_index": 0,
                "chunk_index": 0,
                "total_chunks": 1
            }
        ]

        logger.info("Starting document metadata test...")
        result = self.tree_manager.process_documents(documents, metadata)
        logger.info("Document metadata result: %s", result)

        self.assertIn('tree', result)
        self.assertIn('nodes', result['tree'])
        self.assertEqual(len(result['tree']['nodes']), 1, "Expected one node for the document")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2)