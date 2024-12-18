import os
from typing import List, Dict
import openai
from dotenv import load_dotenv
from src.utils.pinecone_utils import init_pinecone, get_index
from src.utils import config
from src.prepare.embeddings import embed_texts
from src.utils.openai_client import UnifiedAIClient

def get_pinecone_config():
    """Get Pinecone configuration from environment."""
    return {
        'api_key': os.getenv('PINECONE_API_KEY'),
        'environment': os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')
    }

def query_index(query: str, index, client: UnifiedAIClient, top_k: int = 3):
    """
    Query a single index and return results.
    """
    try:
        # Generate embedding for the query
        print("Generating embedding...")
        query_embedding = embed_texts([query])[0]
        
        # Query the index
        print(f"Querying index with vector of size {len(query_embedding)}...")
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        print(f"Got {len(results.matches)} matches")
        
        # Format the results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                'text': match.metadata.get('text', 'No text found'),
                'score': match.score,
                'metadata': {k: v for k, v in match.metadata.items() if k != 'text'}
            })
        
        return formatted_results
    except Exception as e:
        print(f"Error querying index: {str(e)}")
        return []

def inspect_index(name: str, index):
    """
    Inspect an index's statistics and metadata structure.
    """
    print(f"\nInspecting index: {name}")
    print("-" * 80)
    
    try:
        # Get index statistics
        stats = index.describe_index_stats()
        total_vectors = sum(ns.vector_count for ns in stats.namespaces.values())
        print(f"Total vectors: {total_vectors}")
        print(f"Dimension: {stats.dimension}")
        print("\nNamespaces:")
        for ns, ns_stats in stats.namespaces.items():
            print(f"  {ns}: {ns_stats.vector_count} vectors")
        
        # Get a sample of vectors to inspect metadata structure
        sample_results = index.query(
            vector=[0.1] * stats.dimension,  # Use non-zero vector
            top_k=1,
            include_metadata=True
        )
        
        if sample_results.matches:
            print("\nSample metadata structure:")
            for k, v in sample_results.matches[0].metadata.items():
                print(f"  {k}: {type(v).__name__}")
        else:
            print("\nNo vectors found in index")
            
    except Exception as e:
        print(f"Error inspecting index: {str(e)}")

def query_both_indexes(query: str, index1_name: str, index2_name: str, top_k: int = 3):
    """
    Query both indexes and return their results.
    """
    # Initialize AI client
    ai_client = UnifiedAIClient()
    
    # Initialize Pinecone
    pinecone_config = get_pinecone_config()
    print(f"\nInitializing Pinecone with config: {pinecone_config}")
    pinecone = init_pinecone(pinecone_config)
    
    # Query first index
    print(f"\nConnecting to index: {index1_name}")
    index1 = get_index(index1_name)
    inspect_index(index1_name, index1)
    print(f"\nQuerying {index1_name}...")
    results1 = query_index(query, index1, ai_client, top_k)
    
    # Query second index
    print(f"\nConnecting to index: {index2_name}")
    index2 = get_index(index2_name)
    inspect_index(index2_name, index2)
    print(f"\nQuerying {index2_name}...")
    results2 = query_index(query, index2, ai_client, top_k)
    
    return results1, results2

def format_results_for_gpt(results: List[Dict]) -> str:
    """
    Format results in a clear way for GPT analysis.
    """
    formatted = ""
    for i, result in enumerate(results, 1):
        formatted += f"Result {i}:\n"
        formatted += f"Score: {result['score']:.4f}\n"
        formatted += f"Content: {result['text'][:200]}...\n"
        formatted += f"Metadata: {result['metadata']}\n\n"
    return formatted

def analyze_with_gpt(query: str, results1: List[Dict], results2: List[Dict], index1_name: str, index2_name: str) -> str:
    """
    Use GPT-3.5-turbo to analyze and compare the results from both indexes.
    """
    client = UnifiedAIClient()
    
    prompt = f"""
    Please analyze and compare the search results from two different indexes for the query: "{query}"

    Results from {index1_name}:
    {format_results_for_gpt(results1)}

    Results from {index2_name}:
    {format_results_for_gpt(results2)}

    Please provide:
    1. A comparison of the relevance and quality of results from both indexes
    2. Analysis of any notable differences in the retrieved content
    3. Recommendations on which index seems to perform better for this query and why
    """

    response = client.openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at analyzing and comparing search results from different indexes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def print_results(name: str, results: List[Dict]):
    """
    Print formatted results from an index.
    """
    print(f"\nResults from {name}:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text']}")
        if result['metadata']:
            print(f"Metadata: {result['metadata']}")
        print("-" * 40)

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get index names from environment variables
    index1_name = os.getenv("INDEX1_NAME")
    index2_name = os.getenv("INDEX2_NAME")
    
    if not index1_name or not index2_name:
        raise ValueError("Both INDEX1_NAME and INDEX2_NAME environment variables must be set")
    
    # Use predefined query
    query = "Co to jest Vendo?"
    
    print(f"\nQuerying indexes: {index1_name} and {index2_name}...")
    print(f"Query: {query}")
    results1, results2 = query_both_indexes(query, index1_name, index2_name, top_k=4)
    
    # Print raw results from both indexes
    print_results(index1_name, results1)
    print_results(index2_name, results2)
    
    print("\nAnalyzing results with GPT-3.5-turbo...")
    analysis = analyze_with_gpt(query, results1, results2, index1_name, index2_name)
    
    print("\nGPT Analysis:")
    print("=" * 80)
    print(analysis)
    print("=" * 80)

if __name__ == "__main__":
    main()
