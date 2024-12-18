import unittest
from src.embedding.embed_manager import EmbedManager
from unittest.mock import Mock

class TestEmbedManager(unittest.TestCase):
    def setUp(self):
        self.config = {'batch_size': 100}
        self.mock_model = Mock()
        self.manager = EmbedManager(self.mock_model, self.config) 