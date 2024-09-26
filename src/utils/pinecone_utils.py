import time
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from src.utils.config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_DIMENSION,
    PINECONE_METRIC, PINECONE_CLOUD, PINECONE_REGION, PINECONE_NAMESPACE, VERBOSE
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
                index = pc.Index(PINECONE_INDEX_NAME)
                stats = index.describe_index_stats()
                print(f"Index stats: {stats}")
                return index

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
            'id': f'vec_{int(time.time())}_{i}',  # Ensure unique IDs
            'values': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            'metadata': {'text': text}
        }
        if metadata and i < len(metadata):
            vector['metadata'].update(metadata[i])
        vectors.append(vector)
    
    if VERBOSE:
        print(f"Preparing to upsert {len(vectors)} vectors to Pinecone")
        if vectors:
            print(f"Sample vector - ID: {vectors[0]['id']}, Metadata: {vectors[0]['metadata']}, Vector length: {len(vectors[0]['values'])}")
        else:
            print("No vectors to upsert")
    
    if vectors:
        batch_size = 100  # Pinecone recommends batches of 100 vectors
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            try:
                index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)
                if VERBOSE:
                    print(f"Upserted batch of {len(batch)} vectors to Pinecone namespace: {PINECONE_NAMESPACE}")
            except Exception as e:
                print(f"Error upserting batch to Pinecone: {e}")
                print(f"Number of vectors in problematic batch: {len(batch)}")
        
        if VERBOSE:
            print(f"Completed upserting all {len(vectors)} vectors to Pinecone")
    else:
        print("Warning: No vectors to upsert")

def query_pinecone(index, query_embedding, top_k=5):
    # Convert numpy array to list if necessary
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()
    
    if VERBOSE:
        print(f"Querying Pinecone index: {PINECONE_INDEX_NAME}")
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE
    )
    
    if VERBOSE:
        print(f"Pinecone query returned {len(results['matches'])} matches")
        for i, match in enumerate(results['matches']):
            print(f"Match {i+1}: score = {match['score']}, metadata = {match['metadata']}")
    
    return results
