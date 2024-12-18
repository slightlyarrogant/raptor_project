import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import json
from nltk.tokenize import word_tokenize
import nltk
from wordcloud import WordCloud
import os
from src.storage.store_manager import StoreManager
from src.prepare.data_loader import normalize_documents
from typing import List, Dict, Union

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    """Analyzes document corpus and generates visualizations."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """Initialize the document analyzer.
        
        Args:
            output_dir: Directory to save analysis outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            self.logger.warning(f"Failed to download NLTK data: {str(e)}")
            
    def analyze_corpus(self, texts: List[str], metadata: List[Dict]):
        """Analyze document corpus and generate visualizations."""
        try:
            self.logger.info(f"\nAnalyzing corpus of {len(texts)} documents")
            
            # Generate visualizations
            self._generate_length_distribution(texts)
            self._generate_word_cloud(texts)
            self._generate_file_stats(metadata)
            self._generate_content_analysis(texts)
            
            # Save analysis data
            analysis_data = {
                'document_stats': self._calculate_doc_stats(texts),
                'file_stats': self._calculate_file_stats(metadata),
                'content_stats': self._calculate_content_stats(texts)
            }
            
            with open(self.output_dir / 'document_analysis.json', 'w') as f:
                json.dump(analysis_data, f, indent=2)
                
            self.logger.info("✓ Document analysis completed")
            return analysis_data
            
        except Exception as e:
            self.logger.error(f"Document analysis failed: {str(e)}")
            raise
            
    def _generate_length_distribution(self, texts: List[str]):
        """Generate document length distribution visualization."""
        try:
            lengths = [len(text) for text in texts]
            
            plt.figure(figsize=(10, 6))
            sns.histplot(lengths, bins=20)
            plt.title("Document Length Distribution")
            plt.xlabel("Length (characters)")
            plt.ylabel("Count")
            
            # Add statistics
            stats_text = (
                f"Total Documents: {len(texts)}\n"
                f"Mean Length: {np.mean(lengths):.0f}\n"
                f"Median Length: {np.median(lengths):.0f}\n"
                f"Min Length: {min(lengths)}\n"
                f"Max Length: {max(lengths)}"
            )
            plt.text(0.95, 0.95, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.savefig(self.output_dir / 'length_distribution.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to generate length distribution: {str(e)}")
            
    def _generate_word_cloud(self, texts: List[str]):
        """Generate word cloud visualization."""
        try:
            combined_text = " ".join(texts)
            
            wordcloud = WordCloud(
                width=1200,
                height=800,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(combined_text)
            
            plt.figure(figsize=(15, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(self.output_dir / 'word_cloud.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to generate word cloud: {str(e)}")
            
    def _generate_file_stats(self, metadata: List[Dict]):
        """Generate file statistics visualization."""
        try:
            # Count file types
            file_types = Counter(meta.get('extension', 'unknown') for meta in metadata)
            
            plt.figure(figsize=(10, 6))
            plt.bar(file_types.keys(), file_types.values())
            plt.title("File Type Distribution")
            plt.xlabel("File Type")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            
            plt.savefig(self.output_dir / 'file_types.png', bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to generate file stats: {str(e)}")
            
    def _generate_content_analysis(self, texts: List[str]):
        """Generate content analysis visualization."""
        try:
            # Calculate words per document
            words_per_doc = [len(word_tokenize(text)) for text in texts]
            
            plt.figure(figsize=(10, 6))
            sns.histplot(words_per_doc, bins=20)
            plt.title("Words per Document Distribution")
            plt.xlabel("Word Count")
            plt.ylabel("Number of Documents")
            
            plt.savefig(self.output_dir / 'words_per_doc.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to generate content analysis: {str(e)}")
            
    def _calculate_doc_stats(self, texts: List[str]) -> Dict:
        """Calculate document statistics."""
        lengths = [len(text) for text in texts]
        return {
            'total_documents': len(texts),
            'total_characters': sum(lengths),
            'avg_length': np.mean(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths)
        }
        
    def _calculate_file_stats(self, metadata: List[Dict]) -> Dict:
        """Calculate file statistics."""
        file_types = Counter(meta.get('extension', 'unknown') for meta in metadata)
        return {
            'file_types': dict(file_types),
            'total_files': len(metadata)
        }
        
    def _calculate_content_stats(self, texts: List[str]) -> Dict:
        """Calculate content statistics."""
        words_per_doc = [len(word_tokenize(text)) for text in texts]
        return {
            'total_words': sum(words_per_doc),
            'avg_words_per_doc': np.mean(words_per_doc),
            'min_words': min(words_per_doc),
            'max_words': max(words_per_doc)
        }

def run_analysis():
    """Run document analysis pipeline."""
    try:
        # Create output directories
        os.makedirs("analysis_outputs", exist_ok=True)
        os.makedirs("cache", exist_ok=True)
        
        # Initialize components
        analyzer = DocumentAnalyzer(output_dir=Path("analysis_outputs"))
        store_manager = StoreManager()
        
        # Load documents from both databases
        base_path = Path("data")
        doc_paths = [
            base_path / "raptor-technicalbase" / "raw",
            base_path / "raptor-cfi" / "raw"
        ]
        
        texts = []
        metadata = []
        
        # Support multiple file extensions
        extensions = ['.md', '.txt', '.doc', '.docx']
        
        for docs_path in doc_paths:
            logger.info(f"\nScanning directory: {docs_path}")
            
            if not docs_path.exists():
                logger.warning(f"Directory not found: {docs_path}")
                continue
                
            for ext in extensions:
                for file_path in docs_path.glob(f"**/*{ext}"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            texts.append(text)
                            metadata.append({
                                'file_name': file_path.name,
                                'file_path': str(file_path),
                                'file_size': len(text),
                                'extension': ext,
                                'database': 'technical' if 'technicalbase' in str(file_path) else 'cfi'
                            })
                            logger.info(f"Loaded: {file_path.name} ({len(text):,} chars)")
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {str(e)}")
        
        if not texts:
            raise ValueError(f"No documents found in {doc_paths}")
            
        logger.info(f"\nTotal documents loaded: {len(texts)}")
        logger.info(f"From technical base: {sum(1 for m in metadata if m['database'] == 'technical')}")
        logger.info(f"From CFI base: {sum(1 for m in metadata if m['database'] == 'cfi')}")
        
        # Normalize documents
        logger.info("\nNormalizing documents...")
        normalized_texts, normalized_metadata = normalize_documents(texts, metadata)
        logger.info(f"After normalization: {len(normalized_texts)} sections")
        
        # Run analysis
        logger.info("\nRunning document analysis...")
        analyzer.analyze_corpus(normalized_texts, normalized_metadata)
        
        # Log statistics
        logger.info("\nDocument Analysis Results:")
        logger.info("-" * 50)
        logger.info(f"Original documents: {len(texts)}")
        logger.info(f"Normalized sections: {len(normalized_texts)}")
        
        # Log size distribution
        sizes = [len(text) for text in texts]
        logger.info("\nDocument Size Distribution:")
        logger.info(f"Minimum size: {min(sizes):,} chars")
        logger.info(f"Maximum size: {max(sizes):,} chars")
        logger.info(f"Median size: {sorted(sizes)[len(sizes)//2]:,} chars")
        
        logger.info("\nVisualization files generated:")
        logger.info(f"- Length distribution: analysis_outputs/length_distribution.png")
        logger.info(f"- Word cloud: analysis_outputs/word_cloud.png")
        logger.info(f"- File types: analysis_outputs/file_types.png")
        logger.info(f"- Words per doc: analysis_outputs/words_per_doc.png")
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_analysis()