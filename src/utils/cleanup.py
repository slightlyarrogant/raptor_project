import logging
import shutil
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime
import os
from .file_manager import FileManager

logger = logging.getLogger(__name__)

class SystemCleanup:
    """Utility for cleaning up debug files and standardizing file locations."""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.file_manager = FileManager(index_name)
        
        # Debug patterns to clean up
        self.debug_patterns = [
            '*.debug',
            '*.log',
            'debug_*.json',
            'temp_*.json',
            '*.temp',
            '*.bak',
            '*.corrupted'
        ]
        
        # Debug directories to clean
        self.debug_dirs = [
            'debug',
            'temp',
            'logs/debug',
            'cache'
        ]

    def find_debug_files(self) -> List[Path]:
        """Find all debug files in the project."""
        debug_files = []
        
        # Search in all directories
        for pattern in self.debug_patterns:
            for file in Path('.').rglob(pattern):
                if file.is_file():
                    debug_files.append(file)
        
        return debug_files

    def find_debug_dirs(self) -> List[Path]:
        """Find all debug directories in the project."""
        debug_dirs = []
        
        for dir_name in self.debug_dirs:
            for dir_path in Path('.').rglob(dir_name):
                if dir_path.is_dir():
                    debug_dirs.append(dir_path)
        
        return debug_dirs

    def analyze_files(self) -> Dict:
        """Analyze files and their locations."""
        analysis = {
            'debug_files': [],
            'misplaced_files': [],
            'empty_dirs': [],
            'large_files': []
        }
        
        # Find debug files
        debug_files = self.find_debug_files()
        for file in debug_files:
            analysis['debug_files'].append({
                'path': str(file),
                'size': file.stat().st_size,
                'modified': datetime.fromtimestamp(file.stat().st_mtime).isoformat()
            })
        
        # Find misplaced files
        for file in Path('.').rglob('*'):
            if file.is_file():
                # Check if file is in the correct directory based on its type
                if self._is_misplaced(file):
                    analysis['misplaced_files'].append(str(file))
        
        # Find empty directories
        for dir_path in Path('.').rglob('*'):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                analysis['empty_dirs'].append(str(dir_path))
        
        # Find unusually large files (>10MB)
        for file in Path('.').rglob('*'):
            if file.is_file() and file.stat().st_size > 10 * 1024 * 1024:
                analysis['large_files'].append({
                    'path': str(file),
                    'size_mb': file.stat().st_size / (1024 * 1024)
                })
        
        return analysis

    def _is_misplaced(self, file: Path) -> bool:
        """Check if a file is in the wrong directory."""
        # Define file type patterns and their correct directories
        file_patterns = {
            'raw': ['.txt', '.pdf', '.doc', '.docx'],
            'processed': ['.json', '.npy', '.npz'],
            'failed': ['.error.json'],
            'archive': ['.archived']
        }
        
        # Check if file should be in a specific directory
        for dir_type, extensions in file_patterns.items():
            if any(file.name.endswith(ext) for ext in extensions):
                expected_dir = getattr(self.file_manager, f"{dir_type}_dir")
                return not str(file).startswith(str(expected_dir))
        
        return False

    def cleanup_debug_files(self, dry_run: bool = True) -> Dict:
        """Clean up debug files and standardize locations."""
        results = {
            'removed_files': [],
            'removed_dirs': [],
            'moved_files': [],
            'errors': []
        }
        
        try:
            # Find and remove debug files
            debug_files = self.find_debug_files()
            for file in debug_files:
                try:
                    if not dry_run:
                        file.unlink()
                    results['removed_files'].append(str(file))
                except Exception as e:
                    results['errors'].append(f"Failed to remove {file}: {str(e)}")
            
            # Find and remove debug directories
            debug_dirs = self.find_debug_dirs()
            for dir_path in debug_dirs:
                try:
                    if not dry_run:
                        shutil.rmtree(dir_path)
                    results['removed_dirs'].append(str(dir_path))
                except Exception as e:
                    results['errors'].append(f"Failed to remove directory {dir_path}: {str(e)}")
            
            # Move misplaced files to correct locations
            analysis = self.analyze_files()
            for file_path in analysis['misplaced_files']:
                try:
                    file = Path(file_path)
                    if not dry_run:
                        self._move_to_correct_location(file)
                    results['moved_files'].append(str(file))
                except Exception as e:
                    results['errors'].append(f"Failed to move {file}: {str(e)}")
            
            # Remove empty directories
            for dir_path in analysis['empty_dirs']:
                try:
                    if not dry_run:
                        Path(dir_path).rmdir()
                    results['removed_dirs'].append(dir_path)
                except Exception as e:
                    results['errors'].append(f"Failed to remove empty directory {dir_path}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            results['errors'].append(f"General error: {str(e)}")
            return results

    def _move_to_correct_location(self, file: Path) -> None:
        """Move a file to its correct location based on type."""
        # Determine correct location based on file type
        if any(file.name.endswith(ext) for ext in ['.txt', '.pdf', '.doc', '.docx']):
            target_dir = self.file_manager.raw_dir
        elif any(file.name.endswith(ext) for ext in ['.json', '.npy', '.npz']):
            target_dir = self.file_manager.processed_dir
        elif file.name.endswith('.error.json'):
            target_dir = self.file_manager.failed_dir
        elif file.name.endswith('.archived'):
            target_dir = self.file_manager.archive_dir
        else:
            return  # Skip files that don't match any pattern
        
        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Move file to correct location
        target_path = target_dir / file.name
        if target_path.exists():
            # If file already exists, add timestamp to filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            target_path = target_dir / f"{file.stem}_{timestamp}{file.suffix}"
        
        shutil.move(str(file), str(target_path))

def cleanup_system(index_name: str, dry_run: bool = True) -> None:
    """
    Clean up the system by removing debug files and standardizing locations.
    
    Args:
        index_name (str): Name of the index to clean
        dry_run (bool): If True, only show what would be done without making changes
    """
    cleanup = SystemCleanup(index_name)
    
    # First analyze the system
    logger.info("Analyzing system files...")
    analysis = cleanup.analyze_files()
    
    # Print analysis
    logger.info("\nSystem Analysis:")
    logger.info(f"Debug files found: {len(analysis['debug_files'])}")
    logger.info(f"Misplaced files found: {len(analysis['misplaced_files'])}")
    logger.info(f"Empty directories found: {len(analysis['empty_dirs'])}")
    logger.info(f"Large files found: {len(analysis['large_files'])}")
    
    # Confirm before proceeding
    if not dry_run:
        logger.info("\nProceeding with cleanup...")
        results = cleanup.cleanup_debug_files(dry_run=False)
        
        # Print results
        logger.info("\nCleanup Results:")
        logger.info(f"Removed files: {len(results['removed_files'])}")
        logger.info(f"Removed directories: {len(results['removed_dirs'])}")
        logger.info(f"Moved files: {len(results['moved_files'])}")
        if results['errors']:
            logger.warning(f"Errors encountered: {len(results['errors'])}")
            for error in results['errors']:
                logger.warning(error)
    else:
        logger.info("\nDry run - no changes made")
        results = cleanup.cleanup_debug_files(dry_run=True)
        
        # Print what would be done
        logger.info("\nProposed Changes:")
        logger.info("Files to remove:")
        for file in results['removed_files']:
            logger.info(f"  - {file}")
        
        logger.info("\nDirectories to remove:")
        for dir_path in results['removed_dirs']:
            logger.info(f"  - {dir_path}")
        
        logger.info("\nFiles to move:")
        for file in results['moved_files']:
            logger.info(f"  - {file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up system files and standardize locations")
    parser.add_argument("index_name", help="Name of the index to clean")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    cleanup_system(args.index_name, args.dry_run) 