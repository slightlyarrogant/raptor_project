import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.openai_client import UnifiedAIClient
from src.utils.config import CHAT_MODEL, DEFAULT_CONFIG
from src.storage.pinecone_manager import PineconeManager
import tiktoken
import logging
from typing import List, Tuple, Dict
import json
from tqdm import tqdm
import time
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.raw_dir = Path(f"/home/bogdan/Desktop/CFI_Tools/raptor_project/data/{index_name}/raw")
        self.ai_client = UnifiedAIClient()
        self.encoding = tiktoken.get_encoding("cl100k_base")  # Using Claude's tokenizer
        self.model = 'claude-3-5-haiku-20241022'  # Using Claude Haiku explicitly
        
        # Ensure output directory exists
        self.processed_dir = Path(f"/home/bogdan/Desktop/CFI_Tools/raptor_project/data/{index_name}/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset processing status to process all files
        self.status_file = self.processed_dir / "processing_status.json"
        self.status = {
            'processed_files': [],
            'failed_files': [],
            'in_progress': None
        }
        self.save_status()

    def load_status(self):
        """Load or initialize processing status."""
        if self.status_file.exists():
            with open(self.status_file, 'r', encoding='utf-8') as f:
                self.status = json.load(f)
        else:
            self.status = {
                'processed_files': [],
                'failed_files': [],
                'in_progress': None
            }

    def save_status(self):
        """Save current processing status."""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.status, f, indent=2, ensure_ascii=False)

    def get_token_count(self, text: str) -> int:
        """Get token count for text."""
        return len(self.encoding.encode(text))

    def split_text_into_chunks(self, text: str, max_tokens: int = 2000) -> List[str]:
        """Split text into chunks based on token count."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para_tokens = self.get_token_count(para)
            
            if current_length + para_tokens <= max_tokens:
                current_chunk.append(para)
                current_length += para_tokens
            else:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_tokens
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks

    def generate_questions(self, text: str) -> List[str]:
        """Generate relevant questions from text using Claude."""
        prompt = f"""Analyze the following text and generate 3-5 specific, relevant questions that can be answered from the content. 
Focus on key concepts, important details, and main points. Make questions clear and specific.
The questions should start with "Pytanie: ".

Text:
{text}

Generate questions:"""

        try:
            response = self.ai_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.7
            )
            questions = [q.strip() for q in response.split('\n') if q.strip().startswith('Pytanie:')]
            return questions
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []

    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer for a question using Claude."""
        prompt = f"""Based on the following context, provide a clear and concise answer to the question.
The answer should start with "Odpowiedz: ".

Context:
{context}

{question}

Generate answer:"""

        try:
            response = self.ai_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.7
            )
            answer = response.strip()
            if not answer.startswith('Odpowiedz:'):
                answer = f"Odpowiedz: {answer}"
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Odpowiedz: Error generating answer."

    def process_file(self, file_path: Path) -> bool:
        """Process a single file."""
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks if necessary
            chunks = self.split_text_into_chunks(content)
            all_qa_pairs = []
            
            # Process each chunk
            for chunk in chunks:
                # Generate questions
                questions = self.generate_questions(chunk)
                
                # Generate answers for each question
                for question in questions:
                    answer = self.generate_answer(question, chunk)
                    all_qa_pairs.append(f"\n{question}\n{answer}\n")
                
                # Add small delay to avoid rate limits
                time.sleep(1)
            
            # Create Q&A section
            qa_section = "\n\n# Pytania i Odpowiedzi\n" + "\n".join(all_qa_pairs)
            
            # Save processed file
            output_path = self.processed_dir / file_path.name
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content + "\n\n" + qa_section)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return False

    def process_all_files(self):
        """Process all files in the raw directory."""
        try:
            # Get list of files
            files = list(self.raw_dir.glob('*.txt'))
            logger.info(f"Found {len(files)} files to process")
            
            for file_path in tqdm(files):
                if str(file_path) in self.status['processed_files']:
                    logger.info(f"Skipping already processed file: {file_path}")
                    continue
                    
                self.status['in_progress'] = str(file_path)
                self.save_status()
                
                success = self.process_file(file_path)
                
                if success:
                    self.status['processed_files'].append(str(file_path))
                else:
                    self.status['failed_files'].append(str(file_path))
                
                self.status['in_progress'] = None
                self.save_status()
                
            logger.info("Processing completed!")
            logger.info(f"Processed: {len(self.status['processed_files'])} files")
            logger.info(f"Failed: {len(self.status['failed_files'])} files")
            
        except Exception as e:
            logger.error(f"Error in process_all_files: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Process raw files and generate Q&A pairs')
    parser.add_argument('index_name', help='Name of the index to process (e.g., cfi, raptor-cfi)')
    args = parser.parse_args()
    
    processor = DocumentProcessor(args.index_name)
    processor.process_all_files()

if __name__ == "__main__":
    main()
