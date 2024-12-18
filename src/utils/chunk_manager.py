"""Chunk manager for text chunking operations."""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tiktoken
import logging

@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    tokenizer_name: str = "cl100k_base"

@dataclass
class TextChunk:
    """Standard text chunk data model."""
    text: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any]

class ChunkManager:
    """Manager for text chunking operations."""
    
    def __init__(self, config: ChunkConfig):
        """Initialize chunk manager.
        
        Args:
            config: Chunking configuration
        """
        self.config = config
        self.tokenizer = tiktoken.get_encoding(config.tokenizer_name)
        self.logger = logging.getLogger(__name__)
        
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Split text into chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        metadata = metadata or {}
        chunks = []
        start_idx = 0
        
        while start_idx < len(text):
            # Find end of chunk
            end_idx = start_idx + self.config.chunk_size
            if end_idx > len(text):
                end_idx = len(text)
            else:
                # Try to find sentence boundary
                while end_idx > start_idx + self.config.min_chunk_size:
                    if text[end_idx-1] in '.!?':
                        break
                    end_idx -= 1
                    
            # Create chunk
            chunk_text = text[start_idx:end_idx].strip()
            if len(chunk_text) >= self.config.min_chunk_size:
                chunk = TextChunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    metadata={
                        'chunk_idx': len(chunks),
                        'token_count': len(self.tokenizer.encode(chunk_text)),
                        **metadata
                    }
                )
                chunks.append(chunk)
                
            # Move start index for next chunk
            start_idx = end_idx - self.config.chunk_overlap
            if start_idx < 0:
                start_idx = 0
                
        return chunks
        
    def merge_small_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Merge small chunks to meet minimum size requirements.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of merged chunks
        """
        if not chunks:
            return []
            
        merged = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            combined_text = current_chunk.text + " " + next_chunk.text
            combined_tokens = len(self.tokenizer.encode(combined_text))
            
            if combined_tokens <= self.config.max_chunk_size:
                # Merge chunks
                current_chunk = TextChunk(
                    text=combined_text,
                    start_idx=current_chunk.start_idx,
                    end_idx=next_chunk.end_idx,
                    metadata={
                        **current_chunk.metadata,
                        'token_count': combined_tokens,
                        'merged_from': [current_chunk.metadata['chunk_idx'], next_chunk.metadata['chunk_idx']]
                    }
                )
            else:
                # Add current chunk and start new one
                merged.append(current_chunk)
                current_chunk = next_chunk
                
        merged.append(current_chunk)
        return merged
        
    def validate_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Validate chunks meet size requirements.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of valid chunks
        """
        valid_chunks = []
        
        for chunk in chunks:
            token_count = len(self.tokenizer.encode(chunk.text))
            
            if self.config.min_chunk_size <= token_count <= self.config.max_chunk_size:
                chunk.metadata['token_count'] = token_count
                valid_chunks.append(chunk)
            else:
                self.logger.warning(
                    f"Chunk {chunk.metadata['chunk_idx']} size ({token_count} tokens) "
                    f"outside bounds [{self.config.min_chunk_size}, {self.config.max_chunk_size}]"
                )
                
        return valid_chunks
