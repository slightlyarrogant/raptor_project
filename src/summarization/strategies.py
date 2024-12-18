from abc import ABC, abstractmethod
from typing import Dict, List
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import logging
import time
from src.utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)

class SummarizationStrategy(ABC):
    @abstractmethod
    def summarize(self, texts: List[str]) -> str:
        pass

class AbstractiveSummarizer(SummarizationStrategy):
    def __init__(self, config: Dict):
        self.client = OpenAIClient(model=config.get('model'))
        self.max_tokens = config.get('max_tokens', 150)
        self.retry_limit = 3
        self.retry_delay = 2
        self.logger = logger

    def summarize(self, texts: List[str]) -> str:
        """Generate summary for a list of texts."""
        if not texts:
            self.logger.warning("No texts provided for summarization")
            return "No content to summarize"
            
        try:
            combined_text = "\n\n".join(texts)
            prompt = f"Proszę o streszczenie następującego tekstu:\n\n{combined_text}"
            
            response = self.client.generate(prompt)
            self.logger.info("Successfully generated summary")
            return response
            
        except Exception as e:
            self.logger.error(f"Error during summarization: {str(e)}")
            raise

class ExtractiveSummarizer(SummarizationStrategy):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def summarize(self, texts: List[str]) -> str:
        sentences = []
        for text in texts:
            sentences.extend(sent_tokenize(text))
        
        word_freq = self._calculate_word_frequencies(sentences)
        important_sentences = self._select_important_sentences(sentences, word_freq)
        return " ".join(important_sentences)
        
    def _calculate_word_frequencies(self, sentences: List[str]) -> Dict[str, int]:
        """Calculate word frequencies across all sentences."""
        word_freq = Counter()
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [word for word in words if word.isalnum() and word not in self.stop_words]
            word_freq.update(words)
            
        return word_freq
        
    def _select_important_sentences(self, sentences: List[str], word_freq: Dict[str, int], 
                                  num_sentences: int = 3) -> List[str]:
        """Select the most important sentences based on word frequencies."""
        sentence_scores = []
        
        for sentence in sentences:
            score = 0
            words = word_tokenize(sentence.lower())
            words = [word for word in words if word.isalnum()]
            
            for word in words:
                score += word_freq.get(word, 0)
            
            # Normalize by sentence length
            score = score / (len(words) + 1)  # Add 1 to avoid division by zero
            sentence_scores.append((score, sentence))
        
        # Sort by score and select top sentences
        sentence_scores.sort(reverse=True)
        selected_sentences = [sent for _, sent in sentence_scores[:num_sentences]]
        
        # Return sentences in original order
        return [sent for sent in sentences if sent in selected_sentences]