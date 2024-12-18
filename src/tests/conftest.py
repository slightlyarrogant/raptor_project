import os
import sys
from pathlib import Path

# Add project root and src to Python path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / 'src'
sys.path.extend([str(project_root), str(src_path)])
