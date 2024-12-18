import argparse
from src.utils.config import validate_env_vars, VERBOSE
from src.utils.pinecone_utils import initialize_pinecone
from src.prepare.data_loader import load_and_preprocess_documents
from src.prepare.embeddings import create_embedding_model
from src.tree.enhanced_raptor_tree import EnhancedRaptorTree
from src.update.file_manager import move_processed_files, clean_new_documents_folder
from src.inference.rag_chain import create_rag_chain, query_rag_chain
from langchain_openai import ChatOpenAI
from src.utils.config import CHAT_MODEL
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description="RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval")
    parser.add_argument("mode", choices=["prepare", "update", "query", "incremental"], help="Operation mode")
    parser.add_argument("--query", help="Query string for query mode", default=None)
    args = parser.parse_args()

    # Validate environment variables
    validate_env_vars()

    # Initialize components
    vectorstore = initialize_pinecone()
    if vectorstore is None:
        print("Failed to initialize Pinecone. Exiting.")
        return

    embd = create_embedding_model()
    model = ChatOpenAI(temperature=0, model=CHAT_MODEL)

    # Create configuration
    config = {
        'embedding': {'manager': embd},
        'storage': {'manager': vectorstore},
        'index_name': 'raptor-index',
        'chunk_size': 512,
        'overlap_size': 50,
        'max_extension': 100,
        'max_children': 10,
        'min_children': 2,
        'balance_threshold': 0.5,
        'max_depth': 5,
        'clustering': {
            'min_cluster_size': 2,
            'max_cluster_size': 100,
            'min_similarity': 0.05
        }
    }

    # Create EnhancedRaptorTree instance
    raptor_tree = EnhancedRaptorTree(config)

    if args.mode == "prepare":
        if VERBOSE:
            print("Running in preparation mode")
        docs = load_and_preprocess_documents("./data/Raptor_feed")
        
        # Process documents and build tree
        root_node = raptor_tree.process_documents(docs)
        
        # Save tree structure
        tree_path = Path("./data/trees/raptor_tree.json")
        tree_path.parent.mkdir(parents=True, exist_ok=True)
        raptor_tree.save_tree(root_node, str(tree_path))
        
        print("Preparation completed. Tree has been built and saved.")

    elif args.mode == "update":
        if VERBOSE:
            print("Running in update mode")
            
        # Load existing tree if available
        tree_path = Path("./data/trees/raptor_tree.json")
        if tree_path.exists():
            root_node = raptor_tree.load_tree(str(tree_path))
            print("Loaded existing tree structure")
        
        # Process new documents
        new_docs = load_and_preprocess_documents("./data/New_documents")
        updated_root = raptor_tree.process_documents(new_docs)
        
        # Save updated tree
        raptor_tree.save_tree(updated_root, str(tree_path))
        
        # Move processed files
        move_processed_files("./data/New_documents", "./data/Raptor_feed")
        print("Update completed. Tree has been updated and saved.")

    elif args.mode == "query":
        if not args.query:
            print("Please provide a query string with --query")
            return
            
        # Load tree structure
        tree_path = Path("./data/trees/raptor_tree.json")
        if not tree_path.exists():
            print("No tree structure found. Please run in prepare mode first.")
            return
            
        root_node = raptor_tree.load_tree(str(tree_path))
        
        # Find similar nodes
        similar_nodes = raptor_tree.find_similar_nodes(args.query, top_k=5)
        
        print("\nMost relevant document sections:")
        for node, similarity in similar_nodes:
            print(f"\nSimilarity: {similarity:.3f}")
            print(f"Summary: {node.metadata.get('summary', 'No summary available')}")
            print("-" * 50)

    elif args.mode == "incremental":
        if VERBOSE:
            print("Running in incremental mode")
        # Implementation for incremental mode
        pass

    else:
        print(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main()
