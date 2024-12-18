import os
from dotenv import load_dotenv
from src.utils.pinecone_utils import init_pinecone, get_index
import json

def get_pinecone_config():
    """Get Pinecone configuration from environment."""
    return {
        'api_key': os.getenv('PINECONE_API_KEY'),
        'environment': os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')
    }

def fetch_sample_vector(index, index_name):
    """
    Fetch a sample vector from the index with all its details.
    """
    print(f"\nFetching sample vector from {index_name}")
    print("-" * 80)
    
    try:
        # Get index statistics for dimension
        stats = index.describe_index_stats()
        dimension = stats['dimension']
        
        # Query for a single vector
        namespace = None  # we'll try both with and without namespace
        results = index.query(
            vector=[0.0] * dimension,
            top_k=1,
            include_metadata=True,
            include_values=True,  # This is important - we want to see the actual vector
            namespace=namespace
        )
        
        if not results['matches']:
            # Try with namespace if initial attempt failed
            namespace = f"{index_name}_namespace"
            results = index.query(
                vector=[0.0] * dimension,
                top_k=1,
                include_metadata=True,
                include_values=True,
                namespace=namespace
            )
        
        if results['matches']:
            vector = results['matches'][0]
            print(f"Vector ID: {vector['id']}")
            print(f"Score: {vector['score']}")
            print("\nMetadata:")
            print(json.dumps(vector['metadata'], indent=2))
            print("\nFirst 5 vector values:")
            print(vector['values'][:5])
            print(f"Vector dimension: {len(vector['values'])}")
            return vector
        else:
            print("No vectors found in index")
            return None
            
    except Exception as e:
        print(f"Error fetching vector: {str(e)}")
        return None

def main():
    # Load environment variables
    load_dotenv()
    
    # Get index names from environment
    index1_name = os.getenv('INDEX1_NAME')
    index2_name = os.getenv('INDEX2_NAME')
    
    if not (index1_name and index2_name):
        raise ValueError("Both INDEX1_NAME and INDEX2_NAME environment variables must be set")
    
    print(f"Comparing vectors from indexes: {index1_name} and {index2_name}")
    
    # Initialize Pinecone
    pinecone_config = get_pinecone_config()
    print(f"\nInitializing Pinecone with config: {pinecone_config}")
    pinecone = init_pinecone(pinecone_config)
    
    # Get vectors from both indexes
    index1 = get_index(index1_name)
    index2 = get_index(index2_name)
    
    vector1 = fetch_sample_vector(index1, index1_name)
    vector2 = fetch_sample_vector(index2, index2_name)

if __name__ == "__main__":
    main()
