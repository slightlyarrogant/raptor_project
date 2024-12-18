import logging
from typing import List, Dict
from src.utils.openai_client import UnifiedAIClient
import tiktoken
import numpy as np
import time
import signal
from functools import partial
from datetime import datetime
from src.utils.config import DEFAULT_CONFIG, SYSTEM_PROMPTS, SUMMARIZATION_PROMPT

logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

class SummaryManager:
    def __init__(self, config=None, client=None):
        """Initialize summary manager with proper configuration.
        
        Args:
            config: Configuration dictionary
            client: Optional UnifiedAIClient instance. If not provided, a new one will be created.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing SummaryManager...")
        
        # Use DEFAULT_CONFIG and update with any provided config
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
            
        summarization_config = self.config.get('summarization', {})
        
        # Model configuration
        self.model = summarization_config.get('model', DEFAULT_CONFIG['model']['name'])
        self.fallback_model = summarization_config.get('fallback_model', DEFAULT_CONFIG['model']['fallback_name'])
        
        # Initialize or use provided client
        self.client = client if client else UnifiedAIClient()
        
        # Get model info and limits
        self.model_info = self.client.get_model_info(self.model)
        self.max_tokens = min(
            self.model_info['max_output_tokens'],
            summarization_config.get('max_tokens', DEFAULT_CONFIG['model']['max_tokens'])
        )
        self.safe_context_limit = self.model_info['safe_context_limit']
        self.temperature = summarization_config.get('temperature', DEFAULT_CONFIG['model']['temperature'])
        
        # Language configuration
        self.language = summarization_config.get('language', 'pl')  # Default to Polish
        self.supported_languages = summarization_config.get('supported_languages', ['en', 'pl', 'de', 'fr', 'es'])
        
        # Get prompts
        self.system_prompt = SYSTEM_PROMPTS.get('summarization', 
            f"You are a technical documentation expert specializing in {self.language} language summaries. "
            f"Create clear, accurate summaries in {self.language} language only.")
        self.summary_prompt = SUMMARIZATION_PROMPT
        
        # Initialize tokenizer based on model
        self.tokenizer = (
            self.client.anthropic_tokenizer 
            if 'claude' in self.model.lower() 
            else self.client.openai_tokenizer
        )
        
        # Timeout and recovery settings
        self.request_timeout = summarization_config.get('request_timeout', 300)  # 5 minutes
        self.max_retries = summarization_config.get('max_retries', 3)
        self.retry_delay = summarization_config.get('retry_delay', 60)  # 1 minute
        self.last_activity = time.time()
        self.health_check_interval = summarization_config.get('health_check_interval', 1800)  # 30 minutes
        
        # Set up signal handler for timeouts
        signal.signal(signal.SIGALRM, timeout_handler)
        
        self.logger.info(f"SummaryManager initialized with:")
        self.logger.info(f"- Model: {self.model}")
        self.logger.info(f"- Fallback: {self.fallback_model}")
        self.logger.info(f"- Max tokens: {self.max_tokens}")
        self.logger.info(f"- Safe context limit: {self.safe_context_limit}")
        self.logger.info(f"- Language: {self.language}")
        self.logger.info(f"- Request timeout: {self.request_timeout}s")
        self.logger.info(f"- Max retries: {self.max_retries}")
        
        # Test summarization with timeout
        try:
            test_result = self.generate_summary(["Test summary initialization"])
            self.logger.info(f"Generated test summary of {len(test_result)} characters")
            self.logger.info(f"✓ SummaryManager ready with model {self.model}")
        except Exception as e:
            self.logger.error(f"Test summarization failed: {str(e)}")
            self.logger.warning("Proceeding with initialization despite test failure")

    def generate_summary(self, texts: List[str], max_depth: int = 3, additional_prompt: str = None) -> Dict:
        """Generate summary with depth limit to prevent recursion.
        
        Args:
            texts (List[str]): List of texts to summarize
            max_depth (int, optional): Maximum recursion depth. Defaults to 3.
            additional_prompt (str, optional): Additional prompt to guide summarization

        Returns:
            Dict: Summary data
        """
        try:
            if max_depth <= 0:
                return {
                    'title': 'Max depth reached',
                    'summary': ' '.join(texts[:3]),  # Use first 3 texts as fallback
                    'metadata': {'depth': 'max'}
                }
            
            # Track overall time
            start_time = time.time()
            overall_timeout = self.request_timeout * 3  # Triple timeout for full process
            
            combined_text = ' '.join(texts)
            self.logger.info(f"Starting summarization of {len(texts)} texts ({len(combined_text)} chars)")
            
            try:
                # First try direct summarization if text is not too large
                try:
                    token_count = len(self.tokenizer.encode(combined_text))
                    if token_count <= self.safe_context_limit:
                        return self._generate_single_summary(combined_text, additional_prompt)
                except Exception:
                    # If token counting fails, estimate by characters
                    if len(combined_text) <= self.safe_context_limit * 4:  # Rough estimate
                        return self._generate_single_summary(combined_text, additional_prompt)
                
                # If text is too large, split into chunks
                chunks = self._split_text(combined_text)
                self.logger.info(f"Split text into {len(chunks)} chunks")
                
                if len(chunks) > 1:
                    # Process chunks with progress tracking
                    chunk_summaries = []
                    total_chunks = len(chunks)
                    
                    for i, chunk in enumerate(chunks, 1):
                        if time.time() - start_time > overall_timeout:
                            self.logger.warning(f"Overall timeout reached after {i}/{total_chunks} chunks")
                            break
                            
                        self.logger.info(f"Processing chunk {i}/{total_chunks}")
                        chunk_start = time.time()
                        
                        try:
                            # Generate summary for this chunk
                            summary = self._generate_single_summary(chunk, additional_prompt)
                            chunk_summaries.append(summary['summary'])
                            
                            # Log chunk completion
                            chunk_time = time.time() - chunk_start
                            self.logger.info(
                                f"Chunk {i}/{total_chunks} completed in {chunk_time:.1f}s "
                                f"({(i/total_chunks*100):.1f}% done)"
                            )
                            
                        except Exception as e:
                            self.logger.error(f"Error processing chunk {i}: {str(e)}")
                            # Add error placeholder but continue processing
                            chunk_summaries.append(f"Error in chunk {i}: {str(e)}")
                        
                        # Brief pause between chunks to prevent rate limiting
                        time.sleep(0.5)
                    
                    # Combine all chunk summaries into one final summary
                    if not chunk_summaries:
                        raise Exception("No successful chunk summaries generated")
                        
                    self.logger.info(f"Combining {len(chunk_summaries)} chunk summaries")
                    combined_summary = ' '.join(chunk_summaries)
                    
                    # Generate final summary
                    return self._generate_single_summary(combined_summary, additional_prompt)
                
                # If we somehow got here with one chunk, just summarize it
                return self._generate_single_summary(chunks[0], additional_prompt)
                
            finally:
                elapsed = time.time() - start_time
                self.logger.info(f"Total summarization time: {elapsed:.1f}s")
                
        except Exception as e:
            self.logger.error(f"Error in generate_summary: {str(e)}")
            # Return error summary
            return {
                'title': 'Error',
                'summary': f"Failed to generate summary: {str(e)}",
                'metadata': {'error': str(e)}
            }

    def store_summaries(self, store_manager, summaries: List[Dict]) -> None:
        """Store summaries using the store manager."""
        try:
            self.logger.info(f"Storing {len(summaries)} summaries")
            store_manager.store_summaries(summaries)
            self.logger.info("✓ Summaries stored successfully")
        except Exception as e:
            self.logger.error(f"Error storing summaries: {str(e)}")
            raise

    def _generate_single_summary(self, text: str, additional_prompt: str = None) -> Dict:
        """Generate summary for a single piece of text with timeout and fallback handling."""
        import json
        
        # Count tokens
        try:
            # Normalize text encoding first
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
            input_tokens = len(self.tokenizer.encode(text))
        except Exception as e:
            self.logger.warning(f"Token counting failed, using character estimation: {str(e)}")
            input_tokens = len(text) // 4
            
        start_time = time.time()
        self.last_activity = start_time
        
        # Generate summary with timeout and retry logic
        for attempt in range(self.max_retries):
            try:
                # Set alarm for timeout
                signal.alarm(self.request_timeout)
                
                # Format the prompt
                prompt = self.summary_prompt.format(
                    text=text[:1000] + "..." if len(text) > 1000 else text,
                    language=self.language,
                    additional_prompt=additional_prompt if additional_prompt else ""
                )
                
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                current_model = self.model if attempt == 0 else self.fallback_model
                self.logger.info(f"Attempt {attempt + 1}/{self.max_retries} using model: {current_model}")
                
                response = self.client.chat_completion(
                    messages=messages,
                    model=current_model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Clear alarm after successful response
                signal.alarm(0)
                
                self.last_activity = time.time()
                elapsed = time.time() - start_time
                
                # Log the raw response and timing
                self.logger.info(f"Received response in {elapsed:.1f}s ({input_tokens/elapsed:.1f} tokens/s)")
                self.logger.info(f"Raw summarization response:\n{response}")
                
                # Try to parse the response
                summary_data = None
                try:
                    json_str = self.extract_json(response)
                    if json_str:
                        try:
                            summary_data = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            self.logger.error(f"JSON decode error at line {e.lineno}, column {e.colno}")
                            
                        # Convert new format to old format if needed
                        if summary_data and 'summary' not in summary_data and 'main_topic' in summary_data:
                            summary_data = {
                                'title': summary_data.get('title', 'Generated Summary'),
                                'summary': summary_data.get('main_topic', ''),
                                'key_points': [
                                    f"{c['concept']}: {c['description']}"
                                    for c in summary_data.get('key_concepts', [])
                                ] if 'key_concepts' in summary_data else [],
                                'metadata': {
                                    'technical_details': summary_data.get('technical_details', {}),
                                    'relationships': summary_data.get('relationships', []),
                                    'clusters': summary_data.get('clusters', [])
                                }
                            }
                    else:
                        self.logger.warning("Failed to extract JSON from response")
                except Exception as e:
                    self.logger.error(f"Failed to parse response as JSON: {str(e)}")
                
                if not summary_data:
                    # Fallback to using raw response
                    summary_data = {
                        'title': 'Generated Summary',
                        'summary': response.strip(),
                        'metadata': {'parsed': False}
                    }
                
                # Calculate output tokens
                output_tokens = len(self.tokenizer.encode(str(summary_data.get('summary', ''))))
                total_tokens = input_tokens + output_tokens
                
                self.logger.info(f"Summary generated in {elapsed:.2f}s")
                
                return summary_data
                
            except TimeoutError:
                self.logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue
                
            except Exception as e:
                self.logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue
            finally:
                # Always clear the alarm
                signal.alarm(0)
        
        # If all attempts failed
        return {
            'title': 'Error',
            'summary': f"Failed to generate summary after {self.max_retries} attempts",
            'metadata': {'error': 'All attempts failed'}
        }

    def _generate_chunked_summary(self, texts: List[str]) -> Dict:
        """Handle summarization of content exceeding context window with I/O timeout protection."""
        try:
            # Set alarm for overall chunking operation
            signal.alarm(self.request_timeout * 2)  # Double timeout for chunking operations
            
            chunk_summaries = []
            current_chunk = []
            current_tokens = 0
            chunk_size = self.safe_context_limit // 2  # Leave room for prompt and response
            
            self.logger.info(f"Starting chunked summarization with {len(texts)} texts")
            
            for text in texts:
                # Use non-blocking token counting
                try:
                    text_tokens = len(self.tokenizer.encode(text))
                except Exception as e:
                    self.logger.warning(f"Failed to count tokens, estimating based on characters: {str(e)}")
                    text_tokens = len(text) // 4
            
                # If adding this text would exceed chunk size, process current chunk
                if current_tokens + text_tokens > chunk_size and current_chunk:
                    try:
                        # Set shorter timeout for single chunk processing
                        signal.alarm(self.request_timeout)
                        chunk_text = "\n".join(current_chunk)
                        chunk_summary = self._generate_single_summary(chunk_text)
                        chunk_summaries.append(chunk_summary)
                    except Exception as e:
                        self.logger.error(f"Error processing chunk: {str(e)}")
                        # On failure, try processing smaller sub-chunks
                        if len(current_chunk) > 1:
                            mid = len(current_chunk) // 2
                            try:
                                # Process first half
                                sub_chunk1 = "\n".join(current_chunk[:mid])
                                summary1 = self._generate_single_summary(sub_chunk1)
                                chunk_summaries.append(summary1)
                                
                                # Process second half
                                sub_chunk2 = "\n".join(current_chunk[mid:])
                                summary2 = self._generate_single_summary(sub_chunk2)
                                chunk_summaries.append(summary2)
                            except Exception as sub_e:
                                self.logger.error(f"Sub-chunk processing failed: {str(sub_e)}")
                                chunk_summaries.append({
                                    'title': 'Processing Error',
                                    'summary': f'Failed to process chunk due to: {str(e)}',
                                    'metadata': {'error': str(e)}
                                })
                    finally:
                        # Reset for next chunk
                        current_chunk = []
                        current_tokens = 0
                
                # Handle texts larger than chunk size
                if text_tokens > chunk_size:
                    # Split text into smaller pieces (by paragraphs or sentences)
                    import re
                    pieces = text.split('\n\n')  # Split by paragraphs
                    if len(pieces) == 1:  # If no paragraphs, split by sentences
                        pieces = re.split(r'(?<=[.!?])\s+', text)
                    
                    for piece in pieces:
                        try:
                            piece_summary = self._generate_single_summary(piece)
                            chunk_summaries.append(piece_summary)
                        except Exception as e:
                            self.logger.error(f"Error processing piece: {str(e)}")
                else:
                    # Add text to current chunk
                    current_chunk.append(text)
                    current_tokens += text_tokens
            
            # Process any remaining text in the last chunk
            if current_chunk:
                try:
                    chunk_text = "\n".join(current_chunk)
                    chunk_summary = self._generate_single_summary(chunk_text)
                    chunk_summaries.append(chunk_summary)
                except Exception as e:
                    self.logger.error(f"Error processing final chunk: {str(e)}")
            
            # Combine all chunk summaries
            if not chunk_summaries:
                raise ValueError("No successful summaries generated from chunks")
            
            # Generate final summary of summaries if needed
            if len(chunk_summaries) > 1:
                try:
                    summaries = [s.get('summary', '') for s in chunk_summaries if s]
                    final_summary = self._generate_single_summary("\n\n".join(summaries))
                except Exception as e:
                    self.logger.error(f"Failed to generate final summary: {str(e)}")
                    final_summary = {
                        'title': ' | '.join(s.get('title', '') for s in chunk_summaries if s),
                        'summary': '\n\n=== Chunk Break ===\n\n'.join(s.get('summary', '') for s in chunk_summaries if s),
                        'metadata': {
                            'chunked': True,
                            'num_chunks': len(chunk_summaries),
                            'error': str(e)
                        }
                    }
            else:
                final_summary = chunk_summaries[0]
            
            return final_summary
            
        except TimeoutError:
            self.logger.error("Chunking operation timed out")
            raise
        
        except Exception as e:
            self.logger.error(f"Error in chunked summarization: {str(e)}")
            raise
        
        finally:
            # Always clear the alarm
            signal.alarm(0)

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks respecting context limits."""
        try:
            # Get token count
            try:
                total_tokens = len(self.tokenizer.encode(text))
            except Exception as e:
                self.logger.warning(f"Token counting failed, using character estimation: {str(e)}")
                total_tokens = len(text) // 4
            
            # If text fits in one chunk, return as is
            if total_tokens <= self.safe_context_limit:
                return [text]
            
            # Calculate optimal chunk size (leaving room for prompt and response)
            chunk_size = (self.safe_context_limit // 2) - 1000  # Extra buffer for safety
            
            # First try to split by double newlines (paragraphs)
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            paragraphs = text.split('\n\n')
            
            for para in paragraphs:
                para_tokens = len(self.tokenizer.encode(para))
                
                if para_tokens > chunk_size:
                    # If paragraph is too big, split by sentences
                    import re
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    for sent in sentences:
                        sent_tokens = len(self.tokenizer.encode(sent))
                        if sent_tokens > chunk_size:
                            # If sentence is too big, split by words with overlap
                            words = sent.split()
                            current_words = []
                            for word in words:
                                current_words.append(word)
                                if len(self.tokenizer.encode(' '.join(current_words))) >= chunk_size:
                                    chunks.append(' '.join(current_words[:-1]))
                                    current_words = current_words[-100:]  # Keep some overlap
                            if current_words:
                                chunks.append(' '.join(current_words))
                        else:
                            if current_tokens + sent_tokens > chunk_size:
                                chunks.append('\n'.join(current_chunk))
                                current_chunk = [sent]
                                current_tokens = sent_tokens
                            else:
                                current_chunk.append(sent)
                                current_tokens += sent_tokens
                else:
                    if current_tokens + para_tokens > chunk_size:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = [para]
                        current_tokens = para_tokens
                    else:
                        current_chunk.append(para)
                        current_tokens += para_tokens
            
            # Add remaining chunk
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            
            self.logger.info(f"Split {total_tokens} tokens into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error splitting text: {str(e)}")
            # Fallback to simple splitting
            return [text[i:i+5000] for i in range(0, len(text), 5000)]

    def extract_json(self, text: str) -> str:
        """Extract JSON from model response."""
        import re
        import json
        
        # Helper function to try parsing JSON
        def try_parse_json(json_str):
            try:
                # Remove any BOM or hidden characters
                cleaned = json_str.encode('utf-8', 'ignore').decode('utf-8')
                # Remove any non-printable characters except whitespace
                cleaned = ''.join(char for char in cleaned if char.isprintable() or char.isspace())
                # Try to parse
                json.loads(cleaned)
                return cleaned
            except json.JSONDecodeError as e:
                self.logger.debug(f"JSON parse failed: {str(e)}")
                return None
        
        # First try to find JSON between triple backticks
        json_pattern = r'```(?:json)?\s*({\s*".*?"\s*:[\s\S]*?})\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            parsed = try_parse_json(json_str)
            if parsed:
                return parsed
        
        # Try to find any JSON-like structure
        json_pattern = r'{\s*".*?"\s*:[\s\S]*?}'
        match = re.search(json_pattern, text, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            parsed = try_parse_json(json_str)
            if parsed:
                return parsed
                
        # If no valid JSON found in the usual places, try to find any {...} block
        brace_pattern = r'{[\s\S]*}'
        match = re.search(brace_pattern, text, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            parsed = try_parse_json(json_str)
            if parsed:
                return parsed
        
        # If still no valid JSON, log the error
        self.logger.error("Failed to extract valid JSON from response")
        return None

    def summarize_text(self, text: str, additional_prompt: str = None) -> Dict:
        """Generate summary for a piece of text.

        Args:
            text (str): Text to summarize
            additional_prompt (str, optional): Additional prompt to guide summarization

        Returns:
            Dict: Summary data
        """
        if not text:
            return {
                'title': 'Empty Text',
                'summary': '',
                'metadata': {'error': 'Empty input text'}
            }
        
        return self.generate_summary([text], additional_prompt=additional_prompt)

    def summarize(self, text: str) -> str:
        """Generate a summary of the given text."""
        if not self.client:
            raise ValueError("UnifiedAIClient not initialized")
        
        try:
            # Use the existing summarize_text method which handles JSON properly
            summary_data = self.summarize_text(text)
            
            # Return just the summary text from the JSON response
            if isinstance(summary_data, dict):
                return summary_data.get('summary', '')
            return str(summary_data)
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {str(e)}")
            raise
