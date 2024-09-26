import argparse
from src.utils.config import validate_env_vars, VERBOSE
from src.utils.pinecone_utils import initialize_pinecone
from src.prepare.data_loader import load_and_preprocess_documents
from src.prepare.embeddings import create_embedding_model
from src.update.raptor_tree import RaptorTree
from src.update.file_manager import move_processed_files, clean_new_documents_folder
from src.inference.rag_chain import create_rag_chain, query_rag_chain
from langchain_openai import ChatOpenAI
from src.utils.config import CHAT_MODEL

def main():
    parser = argparse.ArgumentParser(description="RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval")
    parser.add_argument("mode", choices=["prepare", "update", "query"], help="Operation mode")
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

    # Create RaptorTree instance
    raptor_tree = RaptorTree(vectorstore, embd, model)

    if args.mode == "prepare":
        if VERBOSE:
            print("Running in preparation mode")
        docs = load_and_preprocess_documents("./data/Raptor_feed")
        raptor_tree.add_new_files(docs)
        print("Preparation completed. Documents have been processed and added to the vector store.")

    elif args.mode == "update":
        if VERBOSE:
            print("Running in update mode")
        new_docs = load_and_preprocess_documents("./data/New_documents")
        raptor_tree.add_new_files(new_docs)
        move_processed_files("./data/New_documents", "./data/Raptor_feed")
        clean_new_documents_folder("./data/New_documents")
        print("Update completed. New documents have been processed and added to the vector store.")

    elif args.mode == "query":
        # Check if the vector store is empty
        stats = vectorstore.describe_index_stats()
        if stats['total_vector_count'] == 0:
            print("The vector store is empty. Please run in 'prepare' mode first to add documents.")
            return

        if args.query is None:
            print("Error: Query string is required in query mode")
            return
        if VERBOSE:
            print("Running in query mode")
        rag_chain = create_rag_chain(vectorstore, embd, model)
        answer = query_rag_chain(rag_chain, args.query)
        print(f"Question: {args.query}")
        print(f"Answer: {answer}")

    else:
        print(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main()
