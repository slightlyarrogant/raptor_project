# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

## Overview

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) is an advanced document processing and retrieval system designed to create a hierarchical knowledge base from a collection of documents. The system uses embedding, clustering, and summarization techniques to organize information in a tree-like structure, allowing for efficient retrieval and querying.

## Features

- Document loading and preprocessing
- Embedding generation using OpenAI's models
- Hierarchical clustering of document chunks
- Recursive summarization of clusters
- Pinecone vector store integration for efficient retrieval
- Retrieval-Augmented Generation (RAG) for question answering
- Versioning and rollback capabilities

## Project Structure

```
raptor_project/
│
├── data/
│   ├── Raptor_feed/          # Existing processed documents
│   └── New_documents/        # Folder for new documents to be added
│
├── src/
│   ├── prepare/
│   │   ├── __init__.py
│   │   ├── data_loader.py    # Functions for loading and preprocessing data
│   │   └── embeddings.py     # Functions for generating embeddings
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── query.py          # Functions for querying the knowledge base
│   │   └── rag_chain.py      # Implementation of the RAG chain
│   │
│   ├── update/
│   │   ├── __init__.py
│   │   ├── raptor_tree.py    # RaptorTree class implementation
│   │   └── file_manager.py   # Functions for managing file movements
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── pinecone_utils.py # Pinecone initialization and management
│   │   └── config.py         # Configuration and constants
│   │
│   └── main.py               # Main script to run the entire process
│
├── tests/
│   ├── test_prepare.py
│   ├── test_inference.py
│   └── test_update.py
│
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/slightlyarrogant/raptor_project.git
   cd raptor_project
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up the `.env` file in the root directory with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_pinecone_index_name
   PINECONE_DIMENSION=1536
   PINECONE_METRIC=cosine
   PINECONE_CLOUD=aws
   PINECONE_REGION=your_pinecone_region
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=your_langchain_api_key
   ```

## Usage

The RAPTOR system can be run in three modes: prepare, update, and query.

1. Preparation mode:
   ```
   python src/main.py prepare
   ```
   This mode processes the documents in the `data/Raptor_feed` folder and builds the initial knowledge base.

2. Update mode:
   ```
   python src/main.py update
   ```
   This mode processes new documents in the `data/New_documents` folder, updates the knowledge base, and moves the processed files to `data/Raptor_feed`.

3. Query mode:
   ```
   python src/main.py query --query "Your question here"
   ```
   This mode allows you to ask questions and receive answers based on the knowledge base.

## Testing

To run the tests, use the following command:
```
python -m unittest discover tests
```

## Contributing

Contributions to RAPTOR are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
