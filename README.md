# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

## Overview

RAPTOR is an advanced document processing and retrieval system designed to create a hierarchical knowledge base from a collection of documents. It uses embedding, clustering, and summarization techniques to organize information in a tree-like structure, allowing for efficient retrieval and querying.

## Features

- Document loading and preprocessing
- Embedding generation using OpenAI's models
- Hierarchical clustering using UMAP and Gaussian Mixture Models
- Recursive summarization of document clusters
- Storage of both original documents and summaries in Pinecone vector store
- Retrieval-Augmented Generation (RAG) for question answering
- Customizable prompts for summarization and querying

## Prerequisites

- Python 3.11.9
- Pinecone account
- OpenAI API key

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/raptor_project.git
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

## Configuration

1. Create a `.env` file in the root directory with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_pinecone_index_name
   PINECONE_DIMENSION=1536
   PINECONE_METRIC=cosine
   PINECONE_CLOUD=aws
   PINECONE_REGION=your_pinecone_region
   PINECONE_NAMESPACE=your_namespace
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=your_langchain_api_key
   CLUSTERING_DIM=10
   CLUSTERING_THRESHOLD=0.1
   ```

2. Customize prompts (optional):
   You can customize the summarization and query prompts by setting these environment variables:
   ```
   SUMMARIZATION_PROMPT="Your custom summarization prompt here"
   QUERY_PROMPT="Your custom query prompt here"
   ```

## Usage

### Preparing Initial Batch of Documents

1. Place your documents in the `data/Raptor_feed` directory.
2. Run the following command:
   ```
   python main.py prepare
   ```

### Querying the System

To query the RAPTOR system:

```
python main.py query --query "Your question here"
```

### Adding New Documents

1. Place the new documents in the `data/New_documents` directory.
2. Run the following command:
   ```
   python main.py update
   ```

## Customizing for Different Use Cases

You can customize RAPTOR for different use cases by modifying the prompts in the `.env` file or by setting environment variables before running the script.

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
│   │   ├── data_loader.py    # Functions for loading and preprocessing data
│   │   └── embeddings.py     # Functions for generating embeddings
│   │
│   ├── inference/
│   │   ├── query.py          # Functions for querying the knowledge base
│   │   └── rag_chain.py      # Implementation of the RAG chain
│   │
│   ├── update/
│   │   ├── raptor_tree.py    # RaptorTree class implementation
│   │   └── file_manager.py   # Functions for managing file movements
│   │
│   ├── utils/
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

## Contributing

Contributions to RAPTOR are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
