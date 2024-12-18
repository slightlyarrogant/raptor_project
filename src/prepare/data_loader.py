import os
from typing import List, Dict, Tuple
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from src.utils.config import CHUNK_SIZE_TOKENS, VERBOSE
import re

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def load_text_files(directory: str) -> List[Document]:
    """Loads all text files from the specified directory."""
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                docs.append(Document(page_content=content, metadata={"filename": filename}))
    return docs

def preprocess_documents(docs: List[Document]) -> List[str]:
    """Concatenate and split documents into chunks."""
    if VERBOSE:
        print("Concatenating and splitting documents")
    
    d_sorted = sorted(docs, key=lambda x: x.metadata["filename"])
    d_reversed = list(reversed(d_sorted))
    concatenated_content = "\n\n\n --- \n\n\n".join([doc.page_content for doc in d_reversed])
    
    if VERBOSE:
        print(f"Num tokens in all context: {num_tokens_from_string(concatenated_content)}")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE_TOKENS, chunk_overlap=0
    )
    texts_split = text_splitter.split_text(concatenated_content)
    
    return texts_split

def load_and_preprocess_documents(directory: str) -> List[str]:
    """Load documents from a directory and preprocess them."""
    if VERBOSE:
        print(f"Loading and processing documents from {directory}")
    
    docs = load_text_files(directory)
    preprocessed_texts = preprocess_documents(docs)
    
    return preprocessed_texts

def normalize_documents(texts: List[str], metadata: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """Normalize documents based on length and content."""
    
    # Calculate average document length
    avg_length = sum(len(text) for text in texts) / len(texts)
    
    normalized_texts = []
    normalized_metadata = []
    
    for i, (text, meta) in enumerate(zip(texts, metadata)):
        # If document is much larger than average, split it into sections
        if len(text) > avg_length * 3:  # Threshold for splitting
            sections = split_large_document(text)
            
            # Create new metadata for each section
            for j, section in enumerate(sections):
                new_meta = meta.copy()
                new_meta.update({
                    'original_doc_id': i,
                    'section_number': j,
                    'total_sections': len(sections),
                    'is_section': True
                })
                normalized_texts.append(section)
                normalized_metadata.append(new_meta)
        else:
            normalized_texts.append(text)
            meta['is_section'] = False
            normalized_metadata.append(meta)
            
    return normalized_texts, normalized_metadata

def split_large_document(text: str) -> List[str]:
    """Split large document into coherent sections."""
    # Try to find natural section breaks first
    sections = []
    
    # Look for common section markers
    markers = [
        r'\n#{1,3}\s+',  # Markdown headers
        r'\n\d+\.\s+',   # Numbered sections
        r'\n[A-Z][^.!?]*[:]\s*\n',  # Title-like lines
        r'\n\s*={3,}\s*\n'  # Separator lines
    ]
    
    # Try to split on natural boundaries
    current_section = []
    lines = text.split('\n')
    
    for line in lines:
        current_section.append(line)
        
        # Check if this line looks like a section boundary
        if any(re.match(pattern, '\n' + line) for pattern in markers):
            if len('\n'.join(current_section)) > 100:  # Minimum section size
                sections.append('\n'.join(current_section))
                current_section = []
    
    # Add any remaining content
    if current_section:
        sections.append('\n'.join(current_section))
    
    # If no natural sections found, fall back to size-based splitting
    if len(sections) <= 1:
        target_size = 2000  # Target characters per section
        sections = []
        current_section = []
        current_size = 0
        
        for line in lines:
            current_section.append(line)
            current_size += len(line)
            
            if current_size >= target_size and line.strip().endswith(('.', '!', '?')):
                sections.append('\n'.join(current_section))
                current_section = []
                current_size = 0
    
    return sections
