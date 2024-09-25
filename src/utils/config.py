import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
RANDOM_SEED = 224
CHUNK_SIZE_TOKENS = 2000
CLUSTERING_DIM = 10
CLUSTERING_THRESHOLD = 0.1
MAX_RECURSION_LEVELS = 3
VERBOSE = True

# API Keys and Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "1536"))
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
PINECONE_REGION = os.getenv("PINECONE_REGION")

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Langchain Configuration
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

def validate_env_vars():
    required_vars = [
        "OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME",
        "PINECONE_DIMENSION", "PINECONE_METRIC", "PINECONE_CLOUD", "PINECONE_REGION"
    ]
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Missing required environment variable: {var}")
