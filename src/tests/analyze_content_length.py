import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.storage.pinecone_manager import PineconeManager
import tiktoken
import logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_token_count(text: str, model: str = None) -> int:
    """Get token count for a text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0

def analyze_index_content(index_name: str):
    """Analyze content length statistics for an index."""
    try:
        # Initialize index
        config = {
            'api_key': os.getenv('PINECONE_API_KEY'),
            'index_name': index_name
        }
        manager = PineconeManager(config)
        
        # Get stats
        stats = manager.index.describe_index_stats()
        total_vectors = sum(ns.vector_count for ns in stats.namespaces.values())
        
        print(f"\nAnalyzing index: {index_name}")
        print("-" * 80)
        print(f"Total vectors: {total_vectors}")
        
        # Initialize stats
        char_lengths = []
        token_lengths = []
        total_chars = 0
        total_tokens = 0
        
        # Query in batches
        batch_size = 100
        for i in range(0, total_vectors, batch_size):
            # Use a normalized vector for querying
            results = manager.index.query(
                vector=[0.1] * stats.dimension,
                top_k=min(batch_size, total_vectors - i),
                include_metadata=True
            )
            
            for match in results.matches:
                text = match.metadata.get('text', '')
                if text:
                    char_len = len(text)
                    token_len = get_token_count(text)
                    
                    char_lengths.append(char_len)
                    token_lengths.append(token_len)
                    total_chars += char_len
                    total_tokens += token_len
        
        # Calculate statistics
        if char_lengths:
            print("\nCharacter Length Statistics:")
            print(f"Total characters: {total_chars:,}")
            print(f"Average characters per vector: {np.mean(char_lengths):.1f}")
            print(f"Median characters per vector: {np.median(char_lengths):.1f}")
            print(f"Min characters: {min(char_lengths):,}")
            print(f"Max characters: {max(char_lengths):,}")
            
            print("\nToken Length Statistics:")
            print(f"Total tokens: {total_tokens:,}")
            print(f"Average tokens per vector: {np.mean(token_lengths):.1f}")
            print(f"Median tokens per vector: {np.median(token_lengths):.1f}")
            print(f"Min tokens: {min(token_lengths):,}")
            print(f"Max tokens: {max(token_lengths):,}")
        
        return {
            'total_vectors': total_vectors,
            'total_chars': total_chars,
            'total_tokens': total_tokens,
            'avg_chars': np.mean(char_lengths) if char_lengths else 0,
            'avg_tokens': np.mean(token_lengths) if token_lengths else 0
        }
            
    except Exception as e:
        logger.error(f"Error analyzing index {index_name}: {str(e)}")
        raise

def main():
    """Compare content length statistics between indexes."""
    try:
        # Set environment
        os.environ['PINECONE_ENVIRONMENT'] = 'aws-us-west-2'
        
        # Analyze both indexes
        cfi_stats = analyze_index_content('cfi')
        raptor_stats = analyze_index_content('raptor-cfi')
        
        # Compare results
        print("\nComparison Summary:")
        print("-" * 80)
        print(f"{'Metric':<30} {'cfi':<20} {'raptor-cfi':<20}")
        print("-" * 80)
        metrics = [
            ('Total Vectors', 'total_vectors', '{:,}'),
            ('Total Characters', 'total_chars', '{:,}'),
            ('Total Tokens', 'total_tokens', '{:,}'),
            ('Avg Characters/Vector', 'avg_chars', '{:.1f}'),
            ('Avg Tokens/Vector', 'avg_tokens', '{:.1f}')
        ]
        
        for name, key, fmt in metrics:
            cfi_val = fmt.format(cfi_stats[key])
            raptor_val = fmt.format(raptor_stats[key])
            print(f"{name:<30} {cfi_val:<20} {raptor_val:<20}")
            
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
