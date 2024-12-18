"""Configuration settings for the Raptor project."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys and Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "1536"))
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
PINECONE_REGION = os.getenv("PINECONE_REGION")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")

# Model configuration
DEFAULT_CONFIG = {
    'model': {
        'name': 'gpt-4o-mini',
        'fallback_name': 'gpt-3.5-turbo',
        'max_tokens': 8000,
        'temperature': 0.3,
        'safe_margin': 0.98
    },
    'chunking': {
        'max_tokens': 4000,  # Safe limit for text-embedding-3-small (8192/2)
        'target_chunk_size': 1000,  # Target size for each chunk
        'overlap_tokens': 100,
        'min_chunk_tokens': 100,
        'max_extension': 50  # Maximum number of tokens to extend a chunk to complete a sentence
    },
    'data': {
        'data_dir': 'data/',
        'supported_extensions': ['.txt', '.md', '.markdown']
    },
    'tree': {
        'max_children': 10,
        'min_children': 2,
        'balance_threshold': 0.5,
    },
    'embedding': {
        'model': 'text-embedding-3-small',
        'dimensions': 1536,
        'batch_size': 8
    },
    'summarization': {
        'model': 'gpt-4o-mini',  # Uses model.name by default
        'fallback_model': 'gpt-3.5-turbo',  # Uses model.fallback_name by default
        'max_tokens': 2000,
        'temperature': 0.3,  # Uses model.temperature by default
        'safe_margin': 0.98,  # Uses model.safe_margin by default
        'language': 'pl',
        'supported_languages': ['en', 'pl', 'de', 'fr', 'es']
    },
    'clustering': {
        'min_cluster_size': 2,
        'max_cluster_size': 10,
        'similarity_threshold': 0.7,
        'algorithm': 'hierarchical',
        'dimension': 15,
        'min_samples': 2,
        'eps': 0.5
    },
    'pinecone': {
        'api_key': PINECONE_API_KEY,
        'dimension': PINECONE_DIMENSION,
        'metric': PINECONE_METRIC,
        'cloud': PINECONE_CLOUD,
        'region': PINECONE_REGION,
        'index_name': PINECONE_INDEX_NAME,
        'namespace': PINECONE_NAMESPACE
    }
}

# Summarization prompt template
SUMMARIZATION_PROMPT = """Analyze and summarize the following technical content. IMPORTANT: Your entire response MUST be in {language} language (all text fields in the JSON must be in {language}).

Content to analyze:
{text}

{additional_prompt}

Provide a comprehensive summary with the following guidelines:
1. For content up to 1000 tokens, provide a summary that is about 20% of the original length
2. For content between 1000-5000 tokens, provide a summary that is about 15% of the original length
3. For content between 5000-10000 tokens, provide a summary that is about 10% of the original length
4. For content over 10000 tokens, provide a summary that is about 5-7% of the original length

Your summary should be at least 2-3 paragraphs long and capture key technical details, relationships, and concepts.

Provide your response in the following JSON format (remember, ALL text must be in {language}):
{{
    "title": "Descriptive title (5-10 words)",
    "summary": "Comprehensive multi-paragraph summary that captures main ideas, technical details, and relationships",
    "key_concepts": [
        {{
            "concept": "Name of concept",
            "description": "Detailed description with technical specifics",
            "importance_score": "1-10 score",
            "technical_details": "Implementation specifics, algorithms, or architectural details",
            "cluster_relevance": "How this concept relates to other items in the cluster",
            "dependencies": ["Related dependencies and requirements"],
            "impact": "Technical impact and implications"
        }}
    ],
    "technical_analysis": {{
        "implementation": {{
            "core_components": ["Detailed description of key components"],
            "algorithms": ["In-depth analysis of algorithms used"],
            "data_structures": ["Comprehensive overview of data structures"],
            "optimizations": ["Performance considerations and optimizations"],
            "technical_debt": ["Potential technical debt or limitations"]
        }},
        "architecture": {{
            "patterns": ["Architectural patterns identified"],
            "decisions": ["Key technical decisions and rationales"],
            "trade_offs": ["Technical trade-offs considered"]
        }},
        "dependencies": {{
            "external": ["External dependencies with versions"],
            "internal": ["Internal component dependencies"],
            "constraints": ["Technical constraints and requirements"]
        }},
        "cluster_patterns": ["Detailed analysis of patterns in this cluster"]
    }},
    "relationships": [
        {{
            "from": "Source component",
            "to": "Target component",
            "type": "Relationship type",
            "description": "Detailed description of the relationship",
            "technical_impact": "Technical implications of this relationship",
            "cluster_context": "How this relationship affects the cluster",
            "dependencies": ["Related dependencies"],
            "constraints": ["Technical constraints"]
        }}
    ],
    "clusters": [
        {{
            "name": "Cluster name",
            "concepts": ["Related concepts"],
            "description": "Detailed technical description",
            "unifying_theme": "Technical characteristics that unify these concepts",
            "distinguishing_features": ["Technical features that make this cluster unique"],
            "implementation_patterns": ["Common implementation patterns"],
            "technical_requirements": ["Specific technical requirements"],
            "optimization_opportunities": ["Potential areas for optimization"]
        }}
    ]
}}

Remember:
1. ALL text in your response must be in {language}
2. Provide detailed technical descriptions while maintaining clarity
3. Include specific implementation details, algorithms, and data structures
4. Analyze patterns at both component and cluster levels
5. Document technical decisions, trade-offs, and rationales
6. Consider performance implications and optimization opportunities
7. Maintain technical accuracy and precision
8. Adapt summary length based on input size (20% for small inputs, scaling down to 5-7% for large inputs)
9. Ensure summaries are at least 2-3 paragraphs long
10. Stay within token limits while maximizing information density"""

# Default system prompts
SYSTEM_PROMPTS = {
    'cluster_summary': (
        "You are a technical documentation expert. Your task is to analyze and summarize "
        "technical content while maintaining key details and technical accuracy. Focus on:\n"
        "1. Main technical concepts and their relationships\n"
        "2. Key implementation details and requirements\n"
        "3. Important technical specifications and parameters\n"
        "4. Critical dependencies and integrations\n"
        "5. Notable technical constraints or limitations\n\n"
        "Provide a clear, structured summary that technical professionals can use for quick reference. "
        "Use the same language as the source documents."
    ),
    
    'general': (
        "You are a helpful AI assistant specializing in technical documentation and knowledge management."
    ),
    
    'query': (
        "You are a technical documentation expert. Your task is to help users find relevant "
        "information in the knowledge base. Focus on:\n"
        "1. Understanding the user's query intent\n"
        "2. Identifying key technical concepts\n"
        "3. Finding relevant documentation sections\n"
        "4. Providing accurate and concise responses\n"
        "5. Including relevant context when needed\n\n"
        "Use the same language as source documents."
    ),
    
    'summarization': (
        "You are a technical documentation expert. Your task is to create a concise yet "
        "comprehensive summary of the provided technical content. The summary should:\n\n"
        "1. Begin with a clear, one-sentence overview of the main topic or purpose\n"
        "2. Identify and explain the key technical components and their relationships\n"
        "3. Highlight critical implementation details, specifications, or requirements\n"
        "4. Note any important dependencies, constraints, or limitations\n"
        "5. Preserve technical accuracy while being accessible\n\n"
        "Format your response as a JSON object with this structure:\n"
        "{\n"
        '  "overview": "One sentence overview",\n'
        '  "key_points": ["Important point 1", "Important point 2", ...],\n'
        '  "technical_details": {\n'
        '    "components": ["Component 1", "Component 2", ...],\n'
        '    "dependencies": ["Dependency 1", "Dependency 2", ...],\n'
        '    "constraints": ["Constraint 1", "Constraint 2", ...]\n'
        '  },\n'
        '  "summary": "Detailed 2-3 sentence technical summary"\n'
        "}\n\n"
        "Keep the response focused, technically accurate, and within 2048 tokens."
    )
}

# Constants
RANDOM_SEED = 224
CHUNK_SIZE_TOKENS = 300  # Reduced from 512 to 300 for finer granularity
CLUSTERING_DIM = int(os.getenv("CLUSTERING_DIM", "10"))
CLUSTERING_THRESHOLD = float(os.getenv("CLUSTERING_THRESHOLD", "0.1"))
MAX_RECURSION_LEVELS = 3
VERBOSE = False

# RAPTOR indices configuration
RAPTOR_INDICES = {
    'raptor-cfi': {
        'dimension': 1536,
        'metric': 'cosine',
        'pods': 1,
        'replicas': 1,
        'shards': 1,
        'metadata_config': {
            'indexed': ['type', 'source', 'section']
        }
    }
}

# Add multiple index configurations
PINECONE_INDICES = {
    'technical_docs': {
        'index_name': 'tech-docs-index',
        'data_dir': 'data/index1',
        'schedule': '0 0 * * 0'  # Weekly on Sunday at midnight (cron format)
    },
    'legal_docs': {
        'index_name': 'legal-docs-index',
        'data_dir': 'data/index2',
        'schedule': '0 0 * * 1'  # Weekly on Monday at midnight
    },
    'research_docs': {
        'index_name': 'research-docs-index',
        'data_dir': 'data/index3',
        'schedule': '0 0 * * 2'  # Weekly on Tuesday at midnight
    }
}

def validate_env_vars():
    required_vars = [
        "OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME",
        "PINECONE_DIMENSION", "PINECONE_METRIC", "PINECONE_CLOUD", "PINECONE_REGION",
        "PINECONE_NAMESPACE"
    ]
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Missing required environment variable: {var}")

# Initialize prompts
QUERY_PROMPT = SYSTEM_PROMPTS['query']
