from src.utils.pinecone_utils import init_pinecone, get_index

def clear_index(index_name: str = 'raptor-cfi'):
    # Initialize Pinecone
    pinecone_config = {
        'api_key': '60c1b174-eecc-4211-a4ea-cae7fcbf61bc',
        'environment': 'gcp-starter'
    }
    init_pinecone(pinecone_config)
    
    # Get index
    index = get_index(index_name)
    
    # Get current stats
    stats = index.describe_index_stats()
    print('Current namespaces:', stats.namespaces)
    
    # Delete each namespace
    for ns in stats.namespaces:
        print(f'Deleting namespace: {ns}')
        index.delete(delete_all=True, namespace=ns)
        
    print('Cleared all namespaces')
    
    # Verify
    stats = index.describe_index_stats()
    print('Final state:', stats.namespaces)

if __name__ == '__main__':
    clear_index()
