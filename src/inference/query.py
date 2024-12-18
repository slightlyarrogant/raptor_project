from typing import List, Dict
from src.prepare.embeddings import embed_texts
from src.utils.pinecone_utils import query_pinecone
from src.utils.config import VERBOSE

def similarity_search(query: str, vectorstore, embd, top_k: int = 5) -> List[Dict]:
    """
    Perform a similarity search for the given query.

    Args:
    query (str): The query string.
    vectorstore: The vector store (Pinecone index) to search in.
    embd: The embedding model to use for encoding the query.
    top_k (int): The number of top results to return.

    Returns:
    List[Dict]: A list of dictionaries containing the top_k most similar documents and their metadata.
    """
    if VERBOSE:
        print(f"Performing similarity search for query: {query}")

    # Generate embedding for the query
    query_embedding = embed_texts([query], embd)[0]  # Take the first (and only) embedding

    # Perform the search
    results = query_pinecone(vectorstore, query_embedding, top_k)

    # Format the results
    formatted_results = []
    for match in results['matches']:
        formatted_results.append({
            'text': match['metadata'].get('text', 'No text found'),
            'score': match['score'],
            'metadata': {k: v for k, v in match['metadata'].items() if k != 'text'}
        })

    if not formatted_results:
        # If no results, add a dummy result to inform the model
        formatted_results.append({
            'text': "No matching documents found in the database.",
            'score': 0,
            'metadata': {}
        })

    if VERBOSE:
        print(f"Number of results: {len(formatted_results)}")
        for i, result in enumerate(formatted_results[:3], 1):  # Print details for top 3 results
            print(f"Result {i}: score = {result['score']:.4f}, text preview = {result['text'][:50]}...")

    return formatted_results


def format_search_results(results: List[Dict]) -> str:
    """
    Format the search results into a readable string.

    Args:
    results (List[Dict]): The search results to format.

    Returns:
    str: A formatted string containing the search results.
    """
    formatted_output = "Search Results:\n\n"
    for i, result in enumerate(results, 1):
        formatted_output += f"{i}. Score: {result['score']:.4f}\n"
        formatted_output += f"   Text: {result['text'][:100]}...\n"
        formatted_output += f"   Metadata: {result['metadata']}\n\n"
    return formatted_output
