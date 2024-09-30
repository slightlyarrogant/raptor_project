# File: tests/test_raptor_tree.py

import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from src.update.raptor_tree import RaptorTree, RaptorTreeError

class TestRaptorTree(unittest.TestCase):

    def setUp(self):
        self.mock_vectorstore = Mock()
        self.mock_embd = Mock()
        self.mock_model = Mock()
        self.raptor_tree = RaptorTree(self.mock_vectorstore, self.mock_embd, self.mock_model)

    def test_add_new_files(self):
        # Test adding new files to the tree
        new_texts = ["Text 1", "Text 2", "Text 3"]
        with patch.object(self.raptor_tree, 'embed_cluster_summarize_texts') as mock_ecs:
            mock_ecs.return_value = (pd.DataFrame(), pd.DataFrame())
            with patch.object(self.raptor_tree, 'build_tree') as mock_build:
                mock_build.return_value = {}
                with patch.object(self.raptor_tree, 'update_vector_store') as mock_uvs:
                    self.raptor_tree.add_new_files(new_texts)
                    mock_ecs.assert_called_once()
                    mock_build.assert_called_once()
                    mock_uvs.assert_called_once()

    def test_merge_summaries(self):
        existing_summary = "This is an existing summary."
        new_summary = "This is a new summary with additional information."
        
        # Mock the embed_documents method to return dummy embeddings
        self.mock_embd.embed_documents.return_value = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6])
        ]
        
        # Mock the predict method of the model
        self.mock_model.predict.return_value = "This is a merged summary."

        merged = self.raptor_tree.merge_summaries(existing_summary, new_summary)
        self.assertEqual(merged, "This is a merged summary.")

    def test_update_embedding(self):
        existing_embedding = [1.0, 0.0, 0.0]
        new_embedding = [0.0, 1.0, 0.0]
        updated_embedding = self.raptor_tree.update_embedding(existing_embedding, new_embedding)
        self.assertAlmostEqual(np.linalg.norm(updated_embedding), 1.0, places=6)

    def test_is_child_of(self):
        # Test child-parent relationship determination
        child_cluster = {'embedding': [1.0, 0.0, 0.0], 'level': 1}
        parent_cluster = {'embedding': [0.9, 0.1, 0.0], 'level': 2}
        self.assertTrue(self.raptor_tree.is_child_of(child_cluster, parent_cluster))

    def test_perform_clustering(self):
        # Test clustering functionality
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        with patch('umap.UMAP') as mock_umap:
            mock_umap.return_value.fit_transform.return_value = embeddings
            with patch('sklearn.mixture.GaussianMixture') as mock_gmm:
                mock_gmm.return_value.predict_proba.return_value = np.array([[1, 0], [0, 1], [0.5, 0.5]])
                clusters = self.raptor_tree.perform_clustering(embeddings, dim=2, threshold=0.5)
                self.assertEqual(len(clusters), len(embeddings))

    def test_load_existing_knowledge(self):
        mock_results = {
            'matches': [
                {'metadata': {'level': 1, 'cluster_id': 'A', 'summary': 'Summary A'}, 'values': [1.0, 0.0]},
                {'metadata': {'level': 2, 'cluster_id': 'B', 'summary': 'Summary B'}, 'values': [0.0, 1.0]}
            ]
        }
        self.mock_vectorstore.query.return_value = mock_results
        with patch.object(self.raptor_tree, '_query_vectorstore_batches', return_value=[mock_results['matches']]):
            knowledge = self.raptor_tree.load_existing_knowledge()
        self.assertIn(1, knowledge)
        self.assertIn(2, knowledge)
        self.assertIn('A', knowledge[1])
        self.assertIn('B', knowledge[2])

    def test_update_vector_store(self):
        updated_tree = {1: {'A': {'summary': 'Summary A', 'embedding': [1.0, 0.0]}}}
        df_clusters = pd.DataFrame({'text': ['Text A'], 'embd': [[0.0, 1.0]]})
        with patch('src.update.raptor_tree.upsert_to_pinecone') as mock_upsert:
            self.raptor_tree.update_vector_store(updated_tree, df_clusters)
            mock_upsert.assert_called_once()

    def test_error_handling(self):
        # Test error handling
        with self.assertRaises(RaptorTreeError):
            with patch.object(self.raptor_tree, 'embed_cluster_summarize_texts', side_effect=Exception("Test error")):
                self.raptor_tree.add_new_files(["Text"])

if __name__ == '__main__':
    unittest.main()
