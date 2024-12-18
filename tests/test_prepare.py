from langchain.schema import Document
from src.tree.tree_manager import TreeManager
import unittest

class TestPrepare(unittest.TestCase):
    def setUp(self):
        self.tree_manager = TreeManager()

    def test_document_processing(self):
        docs_with_special = [
            {'id': '1', 'text': 'Test @#$ content', 'name': 'test1.txt'},
            {'id': '2', 'text': 'Multiple\nline\ntext', 'name': 'test2.txt'}
        ]
        tree_data, tree_stats = self.tree_manager.process_documents(docs_with_special, index_name='test_index')
        self.assertIsNotNone(tree_data)
        self.assertIn('tree', tree_data)
        self.assertIn('nodes', tree_data['tree'])
        self.assertGreater(len(tree_data['tree']['nodes']), 0, "No nodes created in the tree")

if __name__ == '__main__':
    unittest.main()