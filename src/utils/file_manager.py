import logging
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
import tempfile
import os

logger = logging.getLogger(__name__)

class FileManager:
    """Manages file operations and enforces standard directory structure."""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.base_dir = Path('data')
        self.index_dir = self.base_dir / index_name
        
        # Standard directories
        self.raw_dir = self.index_dir / 'raw'
        self.processed_dir = self.index_dir / 'processed'
        self.failed_dir = self.index_dir / 'failed'
        self.archive_dir = self.index_dir / 'archive'
        self.temp_dir = self.index_dir / 'temp'
        
        # Create standard directories
        self._create_directories()
        
        # Cleanup temporary files on initialization
        self.cleanup_temp_files()

    def _create_directories(self) -> None:
        """Create standard directory structure."""
        for dir_path in [
            self.raw_dir,
            self.processed_dir,
            self.failed_dir,
            self.archive_dir,
            self.temp_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_temp_path(self, prefix: str = None) -> Path:
        """Get a temporary file path in the standard temp directory."""
        temp_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        return self.temp_dir / temp_name

    def atomic_write(self, data: Union[Dict, Any], target_file: Path, ensure_dir: bool = True) -> None:
        """Write data atomically using a temporary file."""
        if ensure_dir:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
        temp_file = self.get_temp_path(prefix='atomic_write')
        try:
            with open(temp_file, 'w') as f:
                if isinstance(data, dict):
                    json.dump(data, f, indent=2)
                else:
                    f.write(str(data))
            
            # Atomic rename
            temp_file.replace(target_file)
            
        except Exception as e:
            logger.error(f"Failed to write file {target_file}: {str(e)}")
            if temp_file.exists():
                temp_file.unlink()
            raise
            
    def safe_read(self, file_path: Path, as_json: bool = True) -> Optional[Union[Dict, str]]:
        """Safely read file contents."""
        try:
            if not file_path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return None
                
            with open(file_path, 'r') as f:
                if as_json:
                    return json.load(f)
                return f.read()
                
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {str(e)}")
            return None

    def move_to_failed(self, file_path: Path, error_info: Dict = None) -> Path:
        """Move a file to the failed directory with error information."""
        if not file_path.exists():
            logger.warning(f"Cannot move non-existent file to failed: {file_path}")
            return None
            
        failed_path = self.failed_dir / file_path.name
        error_path = failed_path.with_suffix('.error.json')
        
        try:
            # Move the file
            shutil.move(str(file_path), str(failed_path))
            
            # Write error information
            if error_info:
                self.atomic_write(error_info, error_path)
                
            return failed_path
            
        except Exception as e:
            logger.error(f"Failed to move file to failed directory: {str(e)}")
            return None

    def move_to_archive(self, file_path: Path) -> Path:
        """Move a file to the archive directory."""
        if not file_path.exists():
            logger.warning(f"Cannot archive non-existent file: {file_path}")
            return None
            
        archive_path = self.archive_dir / file_path.name
        
        try:
            shutil.move(str(file_path), str(archive_path))
            return archive_path
            
        except Exception as e:
            logger.error(f"Failed to archive file: {str(e)}")
            return None

    def cleanup_temp_files(self, max_age_hours: int = 24) -> None:
        """Clean up temporary files older than specified age."""
        try:
            current_time = datetime.now().timestamp()
            max_age_seconds = max_age_hours * 3600
            
            for temp_file in self.temp_dir.glob('*'):
                if temp_file.is_file():
                    file_age = current_time - temp_file.stat().st_mtime
                    if file_age > max_age_seconds:
                        temp_file.unlink()
                        logger.info(f"Cleaned up old temp file: {temp_file}")
                        
        except Exception as e:
            logger.error(f"Error during temp file cleanup: {str(e)}")

    def list_failed_files(self) -> Dict[str, Dict]:
        """List all failed files with their error information."""
        failed_files = {}
        
        try:
            for error_file in self.failed_dir.glob('*.error.json'):
                base_name = error_file.stem.replace('.error', '')
                failed_file = self.failed_dir / f"{base_name}{error_file.suffix}"
                
                if failed_file.exists():
                    error_info = self.safe_read(error_file)
                    failed_files[base_name] = {
                        'file_path': str(failed_file),
                        'error_info': error_info,
                        'timestamp': datetime.fromtimestamp(failed_file.stat().st_mtime).isoformat()
                    }
                    
            return failed_files
            
        except Exception as e:
            logger.error(f"Error listing failed files: {str(e)}")
            return {} 