import time
from pinecone import Pinecone, ServerlessSpec
from src.utils.config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_DIMENSION,
    PINECONE_METRIC, PINECONE_CLOUD, PINECONE_REGION, VERBOSE
)

def initialize_pinecone():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if VERBOSE:
            print(f"Attempting to initialize Pinecone index: {PINECONE_INDEX_NAME}")
        
        if PINECONE_INDEX_NAME in pc.list_indexes().names():
            index_info = pc.describe_index(PINECONE_INDEX_NAME)
            if index_info.dimension != PINECONE_DIMENSION:
                if VERBOSE:
                    print(f"Existing index has incorrect dimension. Deleting index {PINECONE_INDEX_NAME}")
                pc.delete_index(PINECONE_INDEX_NAME)
                if VERBOSE:
                    print(f"Deleted index {PINECONE_INDEX_NAME}")
                time.sleep(20)  # Wait for the deletion to complete
            else:
                if VERBOSE:
                    print(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")
                return pc.Index(PINECONE_INDEX_NAME)

        if VERBOSE:
            print(f"Creating new index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )
        if VERBOSE:
            print(f"Created new serverless Pinecone index: {PINECONE_INDEX_NAME}")
        time.sleep(20)  # Wait for the index to be fully initialized
        return pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        return None

def upsert_to_pinecone(index, texts, embeddings, metadata=None):
    vectors = []
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        vector = {
            'id': f'vec_{i}',
            'values': embedding.tolist(),
            'metadata': {'text': text}
        }
        if metadata and i < len(metadata):
            vector['metadata'].update(metadata[i])
        vectors.append(vector)
    
    index.upsert(vectors=vectors)

def query_pinecone(index, query_embedding, top_k=5):
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results
