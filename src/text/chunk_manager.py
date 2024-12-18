"""Manager for handling text chunking."""
import logging
from typing import List, Dict, Optional, Any, Iterator, Tuple
import tiktoken
import re
from nltk.tokenize import sent_tokenize
from src.utils.config import DEFAULT_CONFIG
from src.text.text_chunk import TextChunk

class ChunkManager:
    """Manager for text chunking operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize chunk manager.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
            
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ChunkManager...")
        
        # Get chunking configuration
        chunk_config = self.config.get('chunking', {})
        
        # Configuration with safe defaults
        self.max_tokens = min(
            chunk_config.get('max_tokens', 4000),  # Default to 4000
            8000  # Hard limit to prevent exceeding model context
        )
        self.target_chunk_size = min(
            chunk_config.get('target_chunk_size', 1000),
            self.max_tokens // 2  # Ensure target is at most half of max
        )
        self.overlap_tokens = min(
            chunk_config.get('overlap_tokens', 100),
            self.target_chunk_size // 4  # Limit overlap to 25% of target
        )
        self.min_chunk_tokens = min(
            chunk_config.get('min_chunk_tokens', 100),
            self.target_chunk_size // 2  # Minimum can't be more than half target
        )
        self.max_extension = min(
            chunk_config.get('max_extension', 50),
            self.target_chunk_size // 10  # Limit extension to 10% of target
        )
        
        # Log configuration
        self.logger.info(f"Chunk configuration:")
        self.logger.info(f"- Max tokens: {self.max_tokens}")
        self.logger.info(f"- Target chunk size: {self.target_chunk_size}")
        self.logger.info(f"- Overlap tokens: {self.overlap_tokens}")
        self.logger.info(f"- Min chunk tokens: {self.min_chunk_tokens}")
        self.logger.info(f"- Max extension: {self.max_extension}")
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def process_documents(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> List[TextChunk]:
        """Process multiple documents with proper error handling.
        
        Args:
            texts: List of document texts
            metadata: Optional list of metadata dicts for each document
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        metadata = metadata or [{}] * len(texts)
        
        for i, (text, meta) in enumerate(zip(texts, metadata)):
            try:
                # First try to decode if bytes
                if isinstance(text, bytes):
                    try:
                        text = text.decode('utf-8')
                    except UnicodeDecodeError:
                        # Try with error handling
                        text = text.decode('utf-8', errors='replace')
                        self.logger.warning(f"Document {i}: Had to use replacement characters for invalid UTF-8")
                
                # Clean and validate text
                if not isinstance(text, str):
                    self.logger.error(f"Document {i}: Invalid text type {type(text)}, skipping")
                    continue
                    
                text = self._clean_text(text)
                if not text.strip():
                    self.logger.warning(f"Document {i}: Empty after cleaning, skipping")
                    continue
                
                # Process the document
                doc_chunks = self.chunk_text(text, meta)
                chunks.extend(doc_chunks)
                
            except Exception as e:
                self.logger.error(f"Error processing document {i}: {str(e)}")
                continue
                
        return chunks

    def chunk_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Iterator[Tuple[str, dict]]:
        """Stream document chunks with proper metadata and paragraph awareness.
        
        Args:
            text: Text to split into chunks
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            Iterator of (chunk_text, metadata) tuples
        """
        return self._smart_chunk_document(text, metadata or {})

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Split text into chunks with metadata using smart chunking.
        
        Args:
            text: Text to split into chunks
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of TextChunk objects containing chunk text and metadata
        """
        if not text:
            return []

        # Convert the iterator to a list of TextChunk objects
        chunks = []
        for chunk_text, chunk_metadata in self._smart_chunk_document(text, metadata or {}):
            chunks.append(TextChunk(text=chunk_text, metadata=chunk_metadata))
        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Replace NULL bytes
        text = text.replace('\x00', '')
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if char == '\n' or char == '\t' or char.isprintable())
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text

    def _smart_chunk_document(self, text: str, metadata: dict) -> Iterator[Tuple[str, dict]]:
        """Stream document chunks with proper metadata and paragraph awareness."""
        filename = metadata.get('filename', 'unknown')
        title = self._extract_title(text) or filename
        
        self.logger.info(f"\nChunking file: {filename}")
        
        try:
            # Split into paragraphs first
            paragraphs = []
            for p in text.split('\n\n'):
                p = p.strip()
                if not p:
                    continue
                    
                # Validate paragraph
                try:
                    # Check if we can encode it
                    self.tokenizer.encode(p)
                    paragraphs.append(p)
                except Exception as e:
                    self.logger.warning(f"Skipping invalid paragraph in {filename}: {str(e)}")
                    continue
            
            total_paragraphs = len(paragraphs)
            if total_paragraphs == 0:
                self.logger.warning(f"No valid paragraphs found in {filename}")
                return
            
            current_chunk = []
            current_size = 0
            chunk_index = 0
            
            for i, paragraph in enumerate(paragraphs):
                # Get paragraph size
                try:
                    para_tokens = self.tokenizer.encode(paragraph)
                    para_size = len(para_tokens)
                except Exception as e:
                    self.logger.error(f"Error encoding paragraph {i} in {filename}: {str(e)}")
                    continue
                
                # If single paragraph exceeds chunk size, split it into sentences
                if para_size > self.target_chunk_size:
                    if current_chunk:
                        # First yield the current accumulated chunk
                        yield self._create_chunk_with_metadata(
                            current_chunk, chunk_index, title, metadata, total_paragraphs
                        )
                        chunk_index += 1
                        current_chunk = []
                        current_size = 0
                    
                    # Split large paragraph into sentences
                    sentences = sent_tokenize(paragraph)
                    sentence_chunk = []
                    sentence_size = 0
                    
                    for sentence in sentences:
                        try:
                            sent_tokens = self.tokenizer.encode(sentence)
                            sent_size = len(sent_tokens)
                            
                            if sent_size > self.target_chunk_size:
                                # If single sentence is too large, split on punctuation
                                sub_parts = re.split('[.!?;]', sentence)
                                for part in sub_parts:
                                    part = part.strip()
                                    if not part:
                                        continue
                                    yield self._create_chunk_with_metadata(
                                        [part], chunk_index, title, metadata, total_paragraphs
                                    )
                                    chunk_index += 1
                            elif sentence_size + sent_size > self.target_chunk_size:
                                # Yield current sentence chunk and start new one
                                if sentence_chunk:
                                    yield self._create_chunk_with_metadata(
                                        sentence_chunk, chunk_index, title, metadata, total_paragraphs
                                    )
                                    chunk_index += 1
                                sentence_chunk = [sentence]
                                sentence_size = sent_size
                            else:
                                sentence_chunk.append(sentence)
                                sentence_size += sent_size
                        except Exception as e:
                            self.logger.error(f"Error processing sentence in {filename}: {str(e)}")
                            continue
                    
                    if sentence_chunk:
                        yield self._create_chunk_with_metadata(
                            sentence_chunk, chunk_index, title, metadata, total_paragraphs
                        )
                        chunk_index += 1
                
                # Normal sized paragraph
                elif current_size + para_size > self.target_chunk_size:
                    # Yield current chunk and start new one
                    if current_chunk:
                        yield self._create_chunk_with_metadata(
                            current_chunk, chunk_index, title, metadata, total_paragraphs
                        )
                        chunk_index += 1
                    current_chunk = [paragraph]
                    current_size = para_size
                else:
                    current_chunk.append(paragraph)
                    current_size += para_size
            
            # Yield any remaining chunk
            if current_chunk:
                yield self._create_chunk_with_metadata(
                    current_chunk, chunk_index, title, metadata, total_paragraphs
                )
        
        except Exception as e:
            self.logger.error(f"Error chunking document {filename}: {str(e)}")

    def _create_chunk_with_metadata(self, text_parts: List[str], chunk_index: int, 
                                  title: str, metadata: Dict, total_chunks: int) -> Tuple[str, Dict]:
        """Create a chunk with metadata, ensuring it doesn't exceed token limits."""
        # Join text parts
        text = '\n\n'.join(text_parts)
        
        # Validate chunk size
        if not self._validate_chunk_size(text):
            self.logger.warning(f"Chunk {chunk_index} exceeds token limit, splitting further")
            # Split into smaller chunks if needed
            return self._split_oversized_chunk(text, chunk_index, title, metadata, total_chunks)
        
        # Create metadata
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'title': title,
            'token_count': len(self.tokenizer.encode(text))
        })
        
        return text, chunk_metadata
        
    def _split_oversized_chunk(self, text: str, chunk_index: int, 
                              title: str, metadata: Dict, total_chunks: int) -> Tuple[str, Dict]:
        """Split an oversized chunk into smaller pieces."""
        # Split into sentences
        sentences = sent_tokenize(text)
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(sentence)
            sentence_size = len(sentence_tokens)
            
            if sentence_size > self.target_chunk_size:
                # Split very long sentences on punctuation
                parts = re.split('[.!?;]', sentence)
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    part_tokens = self.tokenizer.encode(part)
                    if len(part_tokens) <= self.target_chunk_size:
                        current_chunk.append(part)
                        current_size += len(part_tokens)
                    if current_size >= self.target_chunk_size:
                        break
            elif current_size + sentence_size > self.target_chunk_size:
                break
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
                
        # Create chunk from collected sentences
        chunk_text = ' '.join(current_chunk)
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'title': title,
            'token_count': current_size,
            'is_split': True
        })
        
        return chunk_text, chunk_metadata

    def _validate_chunk_size(self, text: str) -> bool:
        """Validate that a chunk's token count is within limits.
        
        Args:
            text: Text to validate
            
        Returns:
            bool: True if valid, False if too large
        """
        try:
            tokens = self.tokenizer.encode(text)
            token_count = len(tokens)
            return token_count <= self.max_tokens
        except Exception as e:
            self.logger.error(f"Error validating chunk size: {str(e)}")
            return False
            
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract title from text using various heuristics."""
        # Try to find a title in the first few lines
        first_lines = text.split('\n')[:5]
        
        for line in first_lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
                
            # Look for common title patterns
            if re.match(r'^#+ ', line):  # Markdown heading
                return line.lstrip('#').strip()
            if len(line) < 100 and not line.endswith('.'):  # Short line without period
                return line
                
        # If no clear title found, use first sentence
        sentences = sent_tokenize(text[:1000])  # Look at first 1000 chars
        if sentences:
            return sentences[0]
            
        return None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))
