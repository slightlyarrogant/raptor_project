"""
Utility functions for text processing in RAPTOR.
"""

import logging
import tiktoken
from typing import List, Optional
import re
from src.utils.config import CHUNK_SIZE_TOKENS

logger = logging.getLogger(__name__)

def split_into_chunks(text: str, max_tokens: int = 4000) -> List[str]:
    """Split text into chunks that fit within token limit."""
    if not text:
        return []
        
    enc = tiktoken.get_encoding("cl100k_base")
    
    # First, split by newlines to maintain document structure
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for paragraph in paragraphs:
        # Count tokens in this paragraph
        paragraph_tokens = len(enc.encode(paragraph))
        
        # If single paragraph is too long, split it
        if paragraph_tokens > max_tokens:
            words = paragraph.split()
            temp_chunk = []
            temp_length = 0
            
            for word in words:
                word_tokens = len(enc.encode(word + ' '))
                if temp_length + word_tokens > max_tokens:
                    chunks.append(' '.join(temp_chunk))
                    temp_chunk = [word]
                    temp_length = word_tokens
                else:
                    temp_chunk.append(word)
                    temp_length += word_tokens
            
            if temp_chunk:
                chunks.append(' '.join(temp_chunk))
            continue
            
        # If adding this paragraph would exceed limit, start new chunk
        if current_length + paragraph_tokens > max_tokens:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(chunk_text)
            current_chunk = [paragraph]
            current_length = paragraph_tokens
        else:
            current_chunk.append(paragraph)
            current_length += paragraph_tokens
    
    # Add the last chunk if it exists
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        chunks.append(chunk_text)
    
    # Log chunking results
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def count_tokens(text: str) -> int:
    """Count tokens in text."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def chunk_text(text: str, chunk_size: Optional[int] = None, overlap: int = 0) -> List[str]:
    """
    Split text into chunks of approximately equal size.
    
    Args:
        text (str): Text to split into chunks
        chunk_size (int, optional): Size of each chunk in tokens. Defaults to CHUNK_SIZE_TOKENS
        overlap (int): Number of tokens to overlap between chunks. Defaults to 0
        
    Returns:
        List[str]: List of text chunks
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE_TOKENS
        
    if not text:
        return []
        
    # Approximate token count (rough estimation)
    words = text.split()
    tokens = len(words)
    
    if tokens <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    
    while start < len(words):
        # Calculate end position for current chunk
        end = start + chunk_size
        
        # If this isn't the last chunk, try to break at a sentence boundary
        if end < len(words):
            # Look for sentence boundaries within the last quarter of the chunk
            search_start = max(start + (chunk_size * 3 // 4), start)
            search_text = ' '.join(words[search_start:end])
            
            # Find last sentence boundary
            sentences = re.split(r'[.!?]+\s+', search_text)
            if len(sentences) > 1:
                # Adjust end position to the sentence boundary
                adjustment = len(' '.join(sentences[:-1]).split())
                end = search_start + adjustment
        
        # Create chunk
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        
        # Move start position for next chunk, considering overlap
        start = end - overlap
        
    return chunks

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
        
    # Replace multiple newlines with single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text.strip()

def extract_sections(text: str) -> List[dict]:
    """
    Extract sections from text based on headers.
    
    Args:
        text (str): Text to extract sections from
        
    Returns:
        List[dict]: List of sections with title and content
    """
    # Split text into lines
    lines = text.split('\n')
    sections = []
    current_section = {'title': '', 'content': []}
    
    for line in lines:
        # Check if line is a header (starts with #)
        if re.match(r'^#+\s', line):
            # If we have content in current section, save it
            if current_section['content']:
                sections.append({
                    'title': current_section['title'],
                    'content': '\n'.join(current_section['content'])
                })
            # Start new section
            current_section = {
                'title': line.lstrip('#').strip(),
                'content': []
            }
        else:
            current_section['content'].append(line)
    
    # Add last section if it has content
    if current_section['content']:
        sections.append({
            'title': current_section['title'],
            'content': '\n'.join(current_section['content'])
        })
    
    return sections