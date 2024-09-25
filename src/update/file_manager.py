import os
import shutil
from src.utils.config import VERBOSE

def move_processed_files(source_dir: str, destination_dir: str):
    """
    Move processed files from the source directory to the destination directory.
    
    Args:
    source_dir (str): Path to the directory containing new documents.
    destination_dir (str): Path to the directory where processed documents should be moved.
    """
    if VERBOSE:
        print(f"Moving processed files from {source_dir} to {destination_dir}")
    
    try:
        # Ensure destination directory exists
        os.makedirs(destination_dir, exist_ok=True)
        
        # Get list of files in the source directory
        files = os.listdir(source_dir)
        
        for file in files:
            source_path = os.path.join(source_dir, file)
            destination_path = os.path.join(destination_dir, file)
            
            # Move the file
            shutil.move(source_path, destination_path)
            
            if VERBOSE:
                print(f"Moved {file} to {destination_dir}")
    
    except Exception as e:
        print(f"Error moving files: {e}")

def clean_new_documents_folder(directory: str):
    """
    Remove all files from the specified directory.
    
    Args:
    directory (str): Path to the directory to be cleaned.
    """
    if VERBOSE:
        print(f"Cleaning directory: {directory}")
    
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                if VERBOSE:
                    print(f"Deleted {filename}")
    
    except Exception as e:
        print(f"Error cleaning directory: {e}")
