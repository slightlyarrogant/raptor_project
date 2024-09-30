import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
RANDOM_SEED = 224
CHUNK_SIZE_TOKENS = 2000
CLUSTERING_DIM = int(os.getenv("CLUSTERING_DIM", "10"))  # Default to 10 if not set
CLUSTERING_THRESHOLD = float(os.getenv("CLUSTERING_THRESHOLD", "0.1"))  # Default to 0.1 if not set
MAX_RECURSION_LEVELS = 3
VERBOSE = False

# API Keys and Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "1536"))
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
PINECONE_REGION = os.getenv("PINECONE_REGION")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "raptor_namespace")

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Langchain Configuration
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Customizable Prompts
SUMMARIZATION_PROMPT = os.getenv("SUMMARIZATION_PROMPT", """
You are an AI assistant tasked with summarizing documents. Please provide a concise summary of the following text, 
highlighting the key points and main ideas. The summary should be informative and capture the essence of the document.

Text to summarize:
{text}

Summary:
""")

QUERY_PROMPT = os.getenv("QUERY_PROMPT", """
You are an AI assistant helping to answer questions based on the provided context. Use the following pieces of context 
to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:
""")

def validate_env_vars():
    required_vars = [
        "OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME",
        "PINECONE_DIMENSION", "PINECONE_METRIC", "PINECONE_CLOUD", "PINECONE_REGION"
    ]
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Missing required environment variable: {var}")
