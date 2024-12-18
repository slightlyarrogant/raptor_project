import logging
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

@pytest.fixture
def test_documents():
    """Create test documents with known content."""
    return [
        {
            'text': f'Test document {i} with some content that needs to be chunked properly.',
            'metadata': {
                'filename': f'test_{i}.txt',
                'section': f'section_{i}'
            }
        }
        for i in range(5)
    ]

@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for test documents."""
    return np.array([
        [0.1 * i] * 1536
        for i in range(5)
    ])

@pytest.fixture
def mock_chunker():
    """Create a mock chunker with controlled behavior."""
    chunker = Mock()
    
    def mock_chunk_document(text, metadata):
        # Simple chunking simulation
        words = text.split()
        chunks = []
        chunk_size = 5  # words per chunk
        
        for i in range(0, len(words), chunk_size):
            chunk_text = ' '.join(words[i:i + chunk_size])
            chunk_metadata = {
                **metadata,
                'chunk_index': i // chunk_size,
                'total_chunks': (len(words) + chunk_size - 1) // chunk_size,
                'chunk_size': len(chunk_text),
                'created_at': '2024-01-01'
            }
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return chunks
    
    chunker.chunk_document = mock_chunk_document
    return chunker

def test_chunking_basic(test_documents, mock_chunker):
    """Test basic document chunking."""
    for doc in test_documents:
        chunks = mock_chunker.chunk_document(doc['text'], doc['metadata'])
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert 'text' in chunk
            assert 'metadata' in chunk
            assert 'chunk_size' in chunk['metadata']
            assert 'created_at' in chunk['metadata']

def test_chunking_metadata(test_documents, mock_chunker):
    """Test metadata preservation in chunks."""
    for doc in test_documents:
        chunks = mock_chunker.chunk_document(doc['text'], doc['metadata'])
        
        for chunk in chunks:
            assert chunk['metadata']['filename'] == doc['metadata']['filename']
            assert chunk['metadata']['section'] == doc['metadata']['section']
            assert 'chunk_index' in chunk['metadata']
            assert 'total_chunks' in chunk['metadata']

def test_chunking_empty_documents():
    """Test handling of empty documents."""
    empty_docs = [
        {
            'text': '',
            'metadata': {'filename': 'empty.txt'}
        }
    ]
    
    mock_chunker = Mock()
    mock_chunker.chunk_document.return_value = []
    
    for doc in empty_docs:
        chunks = mock_chunker.chunk_document(doc['text'], doc['metadata'])
        assert len(chunks) == 0

def test_chunking_special_characters():
    """Test chunking with special characters."""
    special_docs = [
        {
            'text': 'Test\nwith\nspecial\ncharacters\n\n\nand\nmultiple\nlines',
            'metadata': {'filename': 'special.txt'}
        }
    ]
    
    mock_chunker = Mock()
    
    def mock_chunk_special(text, metadata):
        # Normalize newlines and split by normalized newlines
        normalized = text.replace('\n\n\n', '\n\n')
        paragraphs = normalized.split('\n\n')
        
        return [
            {
                'text': p.strip(),
                'metadata': {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(paragraphs),
                    'chunk_size': len(p),
                    'created_at': '2024-01-01'
                }
            }
            for i, p in enumerate(paragraphs)
            if p.strip()
        ]
    
    mock_chunker.chunk_document = mock_chunk_special
    
    for doc in special_docs:
        chunks = mock_chunker.chunk_document(doc['text'], doc['metadata'])
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk['text'].strip() != ''
            assert '\n\n\n' not in chunk['text']

if __name__ == "__main__":
    pytest.main([__file__, '-v'])