import os
import logging
from typing import List, Dict, Union
import openai
from anthropic import Anthropic, RateLimitError
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import tiktoken
import asyncio
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import numpy as np

logger = logging.getLogger(__name__)

class UnifiedAIClient:
    """Unified client for OpenAI and Anthropic API interactions."""
    
    def __init__(self):
        """Initialize clients for both providers."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.openai_tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize Anthropic
        self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # Rate limiting configuration with more conservative limits
        self.rate_limits = {
            'anthropic': {
                'tokens_per_minute': {
                    'claude-3-5-haiku-20241022': 40000,
                    'claude-3-5-sonnet-20241022': 50000
                },
                'requests_per_minute': 5,  # More conservative
                'retry_after': 120,  # Wait 2 minutes by default
                'backoff_multiplier': 2,  # Double wait time after each retry
                'max_retries': 8  # Allow more retries with longer waits
            },
            'openai': {
                'tokens_per_minute': 90000,
                'requests_per_minute': 500,
                'retry_after': 20
            }
        }
        
        # Model configurations with context and output limits
        self.openai_models = {
            'gpt-4o-mini': {
                'max_context_tokens': 128000,
                'max_output_tokens': 4096,
                'safe_context_limit': 100000,
                'is_openai': True
            },
            'gpt-3.5-turbo': {
                'max_context_tokens': 8192,
                'max_output_tokens': 1500,
                'safe_context_limit': 6000,
                'is_openai': True
            },
        }
        
        self.anthropic_models = {
            'claude-3-5-sonnet-20241022': {
                'max_context_tokens': 200000,
                'max_output_tokens': 4096,
                'safe_context_limit': 150000,
                'is_openai': False
            },
            'claude-3-5-haiku-20241022': {
                'max_context_tokens': 200000,
                'max_output_tokens': 8192,
                'safe_context_limit': 150000,
                'is_openai': False
            }
        }
        
        # Initialize tokenizers
        self.anthropic_tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.logger.info("Unified AI client initialized")

    def _check_token_limits(self, text: str, model: str) -> bool:
        """Check if text is within token limits for model."""
        model_config = self.get_model_info(model)
        tokenizer = self.anthropic_tokenizer if 'claude' in model.lower() else self.openai_tokenizer
        token_count = len(tokenizer.encode(text))
        
        if token_count > model_config['safe_context_limit']:
            self.logger.warning(
                f"Text exceeds safe token limit for {model}. "
                f"Tokens: {token_count}, Limit: {model_config['safe_context_limit']}"
            )
            return False
        return True

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.openai_tokenizer.encode(text))

    async def _make_request_with_backoff(self, request_fn, *args, **kwargs):
        """Make a request with exponential backoff for rate limits.
        
        Args:
            request_fn: The OpenAI API function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The API response
            
        Raises:
            Exception: If max retries exceeded
        """
        max_retries = 5
        base_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                return await request_fn(*args, **kwargs)
            except openai.RateLimitError as e:
                if attempt == max_retries - 1:
                    raise  # Re-raise if we're out of retries
                
                # Calculate delay with exponential backoff and jitter
                delay = (2 ** attempt) * base_delay + random.uniform(0, 0.1)
                
                self.logger.warning(f"Rate limit hit. Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
            except Exception as e:
                raise  # Re-raise other exceptions immediately

    async def create_chat_completion(self, *args, **kwargs):
        """Create a chat completion with automatic retries for rate limits."""
        return await self._make_request_with_backoff(
            self.openai_client.chat.completions.create,
            *args,
            **kwargs
        )

    async def create_embedding(self, *args, **kwargs):
        """Create embeddings with automatic retries for rate limits."""
        return await self._make_request_with_backoff(
            self.openai_client.embeddings.create,
            *args,
            **kwargs
        )

    def chat_completion(self, messages: List[Dict], model: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Send a chat completion request to the API with progress tracking."""
        try:
            self.logger.info(f"Starting API request to {model}...")
            start_time = time.time()
            last_progress = start_time
            progress_interval = 2  # Show progress every 2 seconds
            
            def progress_callback():
                nonlocal last_progress
                current_time = time.time()
                if current_time - last_progress >= progress_interval:
                    elapsed = current_time - start_time
                    self.logger.info(f"Still waiting for response... ({elapsed:.1f}s elapsed)")
                    last_progress = current_time
            
            # Set up the request with progress tracking
            headers = {
                "Authorization": f"Bearer {self.openai_client.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Create session for better connection handling
            with requests.Session() as session:
                # Configure session with retry strategy
                retry_strategy = Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                )
                adapter = HTTPAdapter(max_retries=retry_strategy)
                session.mount("https://", adapter)
                
                # Send request with streaming to enable progress updates
                with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    stream=True,
                    timeout=(10, 300)  # (connect timeout, read timeout)
                ) as response:
                    response.raise_for_status()
                    
                    # Read response with progress updates
                    content = ""
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            content += chunk.decode('utf-8')
                            progress_callback()
                    
                    # Parse the response
                    try:
                        result = json.loads(content)
                        completion = result['choices'][0]['message']['content']
                        
                        # Log completion stats
                        duration = time.time() - start_time
                        tokens_per_second = len(completion.split()) / duration
                        self.logger.info(f"Received response in {duration:.1f}s ({tokens_per_second:.1f} tokens/s)")
                        
                        return completion
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        self.logger.error(f"Failed to parse API response: {str(e)}")
                        self.logger.debug(f"Raw response: {content}")
                        raise
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise

    def get_embeddings(self, texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
        """Generate embeddings using OpenAI (Anthropic doesn't support embeddings yet)."""
        try:
            start_time = time.time()
            
            # Handle rate limits with retries
            max_retries = 5
            base_delay = 1
            
            for attempt in range(max_retries):
                try:
                    response = self.openai_client.embeddings.create(
                        model=model,
                        input=texts
                    )
                    
                    duration = time.time() - start_time
                    self.logger.info(f"OpenAI Embeddings: {model} completed in {duration:.2f}s")
                    
                    # Convert embeddings to numpy array for consistency
                    embeddings = [embedding.embedding for embedding in response.data]
                    embeddings_array = np.array(embeddings)
                    
                    # Log shape for debugging
                    self.logger.info(f"Generated embeddings array of shape: {embeddings_array.shape}")
                    
                    return embeddings_array
                    
                except openai.RateLimitError as e:
                    if attempt == max_retries - 1:
                        raise  # Re-raise if we're out of retries
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = (2 ** attempt) * base_delay + random.uniform(0, 0.1)
                    self.logger.warning(f"Rate limit hit. Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    
                except Exception as e:
                    self.logger.error(f"OpenAI embeddings failed: {str(e)}")
                    raise
                    
        except Exception as e:
            self.logger.error(f"Embeddings generation failed: {str(e)}")
            raise

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=60, min=60, max=300),  # Start with 1 minute, max 5 minutes
        stop=stop_after_attempt(8),  # More attempts with longer waits
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    def _anthropic_chat(self, messages: List[Dict], model: str, **kwargs) -> str:
        """Handle Anthropic chat completion with improved rate limit handling."""
        try:
            start_time = time.time()
            
            # Estimate token count
            combined_text = "\n".join(m['content'] for m in messages)
            estimated_tokens = len(self.anthropic_tokenizer.encode(combined_text))
            
            # Get model-specific rate limit
            model_limit = self.rate_limits['anthropic']['tokens_per_minute'].get(
                model, 
                40000  # Default to most conservative limit
            )
            
            # Check if we might exceed rate limits
            if estimated_tokens > model_limit:
                wait_time = self.rate_limits['anthropic']['retry_after']
                self.logger.info(
                    f"Large request ({estimated_tokens} tokens) detected. "
                    f"Waiting {wait_time}s before proceeding..."
                )
                time.sleep(wait_time)
            
            try:
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=kwargs.get('max_tokens', self.anthropic_models[model]['max_output_tokens']),
                    temperature=kwargs.get('temperature', 0.7),
                    messages=[{"role": "user", "content": combined_text}]
                )
                
                duration = time.time() - start_time
                self.logger.info(f"Anthropic Request: {model} completed in {duration:.2f}s")
                
                return response.content[0].text
                
            except RateLimitError as e:
                # Extract retry-after if available
                retry_after = getattr(e, 'retry_after', None)
                if retry_after:
                    wait_time = int(retry_after) + 5  # Add buffer
                else:
                    wait_time = self.rate_limits['anthropic']['retry_after']
                
                self.logger.warning(
                    f"Rate limit hit for {model}, waiting {wait_time}s. "
                    f"Token count: {estimated_tokens}"
                )
                
                # Update rate limit for next time
                self.rate_limits['anthropic']['retry_after'] = min(
                    wait_time * self.rate_limits['anthropic']['backoff_multiplier'],
                    300  # Max 5 minutes
                )
                
                raise  # Let retry decorator handle it
                
        except Exception as e:
            if not isinstance(e, RateLimitError):
                self.logger.error(f"Anthropic chat failed: {str(e)}")
            raise

    def get_model_info(self, model: str) -> Dict:
        """Get model configuration information."""
        if model in self.openai_models:
            return self.openai_models[model]
        elif model in self.anthropic_models:
            return self.anthropic_models[model]
        else:
            raise ValueError(f"Unknown model: {model}")
