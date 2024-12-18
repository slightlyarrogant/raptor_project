import logging
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class CacheCleanup:
    """Utility for cleaning up cache files."""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.data_dir = Path('data') / index_name
        self.cache_dir = self.data_dir / 'cache'
        self.backup_dir = self.data_dir / 'backups'
        
        # Create directories if they don't exist
        for dir in [self.cache_dir, self.backup_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.cache_db = self.cache_dir / 'cache.db'
        self.processing_db = self.cache_dir / 'processing.db'
        self.tree_debug = self.cache_dir / 'tree_debug.json'
        
        logger.info(f"Initialized cache cleanup for index {index_name}")

    def analyze_cache(self) -> Dict:
        """Analyze current cache state."""
        return {
            'cache_db': self._analyze_cache_db(),
            'processing_db': self._analyze_processing_db(),
            'tree_debug': self._analyze_tree_debug()
        }

    def _analyze_cache_db(self) -> Dict:
        """Analyze the cache database."""
        try:
            if not self.cache_db.exists():
                return {'exists': False}
            
            return {
                'exists': True,
                'size': self.cache_db.stat().st_size,
                'last_modified': datetime.fromtimestamp(self.cache_db.stat().st_mtime)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cache db: {str(e)}")
            return {'error': str(e)}

    def _analyze_processing_db(self) -> Dict:
        """Analyze the processing database."""
        try:
            if not self.processing_db.exists():
                return {'exists': False}
            
            return {
                'exists': True,
                'size': self.processing_db.stat().st_size,
                'last_modified': datetime.fromtimestamp(self.processing_db.stat().st_mtime)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing processing db: {str(e)}")
            return {'error': str(e)}

    def _analyze_tree_debug(self) -> Dict:
        """Analyze the tree debug JSON file."""
        try:
            if not self.tree_debug.exists():
                return {'exists': False}
            
            with open(self.tree_debug) as f:
                debug_data = json.load(f)
            
            return {
                'exists': True,
                'size': self.tree_debug.stat().st_size,
                'content': debug_data
            }
            
        except Exception as e:
            logger.error(f"Error analyzing tree debug: {str(e)}")
            return {'error': str(e)}

    def backup_cache_files(self) -> str:
        """Create backup of current cache files."""
        backup_dir = self.backup_dir / f"cache_{int(datetime.now().timestamp())}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Backup each file if it exists
            for file_path in [self.cache_db, self.processing_db, self.tree_debug]:
                if file_path.exists():
                    shutil.copy2(file_path, backup_dir / file_path.name)
            
            logger.info(f"Created backup in: {backup_dir}")
            return str(backup_dir)
            
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            raise

    def clean_processing_db(self) -> None:
        """Clean the processing database."""
        try:
            if self.processing_db.exists():
                self.processing_db.unlink()
                logger.info("Cleaned processing database")
            else:
                logger.info("Processing database does not exist")
                
        except Exception as e:
            logger.error(f"Error cleaning processing db: {str(e)}")
            raise

    def extract_valid_embeddings(self) -> Tuple[List[str], List[List[float]]]:
        """Extract valid text-embedding pairs from cache."""
        try:
            if not self.cache_db.exists():
                logger.warning("Cache database does not exist")
                return [], []
            
            # Implementation depends on cache db structure
            # This is a placeholder that should be implemented based on actual db schema
            return [], []
            
        except Exception as e:
            logger.error(f"Error extracting embeddings: {str(e)}")
            return [], []

def print_analysis_report(stats: Dict) -> None:
    """Print a formatted analysis report."""
    print("\nCache Analysis Report")
    print("=" * 50)
    
    for component, data in stats.items():
        print(f"\n{component.upper()}")
        print("-" * 30)
        
        if 'error' in data:
            print(f"Error: {data['error']}")
            continue
            
        if not data['exists']:
            print("Status: Does not exist")
            continue
            
        print(f"Status: Exists")
        print(f"Size: {data['size'] / 1024:.2f} KB")
        
        if 'last_modified' in data:
            print(f"Last Modified: {data['last_modified']}")
            
        if 'content' in data:
            print("Content available")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Get index name from command line
    import sys
    if len(sys.argv) != 2:
        print("Usage: python cache_cleanup.py <index_name>")
        sys.exit(1)
        
    index_name = sys.argv[1]
    cleanup = CacheCleanup(index_name)
    
    # Analyze current state
    print("\nAnalyzing cache files...")
    stats = cleanup.analyze_cache()
    print_analysis_report(stats)
    
    # Create backup before any modifications
    print("\nCreating backup...")
    backup_path = cleanup.backup_cache_files()
    print(f"Backup created at: {backup_path}")
    
    # Ask for confirmation before cleaning
    response = input("\nWould you like to clean the processing database? (y/n): ")
    if response.lower() == 'y':
        cleanup.clean_processing_db()
        print("Processing database cleaned")
        
        # Extract valid embeddings
        print("\nExtracting valid embeddings...")
        texts, embeddings = cleanup.extract_valid_embeddings()
        print(f"Found {len(texts)} valid text-embedding pairs")
