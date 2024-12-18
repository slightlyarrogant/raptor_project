import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class DocumentManifest:
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.manifest_dir = Path(f"data/{index_name}")
        self.manifest_file = self.manifest_dir / "manifest.json"
        self.manifest: Dict = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """Load or create manifest file."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading manifest: {e}")
                return self._create_default_manifest()
        return self._create_default_manifest()

    def _create_default_manifest(self) -> Dict:
        """Create default manifest structure."""
        return {
            "index_name": self.index_name,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "documents": {},
            "stats": {
                "total_documents": 0,
                "processed_documents": 0,
                "failed_documents": 0,
                "total_chunks": 0,
                "total_nodes": 0
            },
            "version_history": []
        }

    def _save_manifest(self):
        """Save manifest to file."""
        try:
            self.manifest["last_updated"] = datetime.now().isoformat()
            self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_file, 'w') as f:
                json.dump(self.manifest, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving manifest: {e}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def register_document(self, doc_path: Path, metadata: Dict) -> Dict:
        """Register a new document or update existing one."""
        doc_id = str(doc_path)
        file_hash = self._compute_file_hash(doc_path)
        
        doc_entry = {
            "path": str(doc_path),
            "hash": file_hash,
            "original_hash": file_hash,
            "first_processed": datetime.now().isoformat(),
            "last_processed": datetime.now().isoformat(),
            "processing_status": "pending",
            "metadata": metadata,
            "chunks": [],
            "tree_nodes": [],
            "processing_history": [],
            "version": "1.0"
        }
        
        if doc_id in self.manifest["documents"]:
            # Update existing entry
            existing = self.manifest["documents"][doc_id]
            doc_entry["first_processed"] = existing.get("first_processed", doc_entry["first_processed"])
            doc_entry["original_hash"] = existing.get("original_hash", file_hash)
            doc_entry["version"] = f"1.{len(existing.get('processing_history', [])) + 1}"
            
            # Archive previous processing attempt
            existing["processing_history"].append({
                "timestamp": datetime.now().isoformat(),
                "status": existing.get("processing_status", "unknown"),
                "hash": existing.get("hash", ""),
                "version": existing.get("version", "1.0")
            })
            doc_entry["processing_history"] = existing["processing_history"]
        
        self.manifest["documents"][doc_id] = doc_entry
        self._save_manifest()
        return doc_entry

    def update_document_status(self, doc_path: Path, status: str, 
                             chunk_ids: Optional[List[str]] = None,
                             node_ids: Optional[List[str]] = None):
        """Update document processing status and related IDs."""
        doc_id = str(doc_path)
        if doc_id in self.manifest["documents"]:
            doc = self.manifest["documents"][doc_id]
            doc["processing_status"] = status
            doc["last_processed"] = datetime.now().isoformat()
            
            if chunk_ids:
                doc["chunks"] = chunk_ids
            if node_ids:
                doc["tree_nodes"] = node_ids
                
            # Update stats
            stats = self.manifest["stats"]
            if status == "processed":
                stats["processed_documents"] = stats.get("processed_documents", 0) + 1
                if chunk_ids:
                    stats["total_chunks"] = stats.get("total_chunks", 0) + len(chunk_ids)
                if node_ids:
                    stats["total_nodes"] = stats.get("total_nodes", 0) + len(node_ids)
            elif status == "failed":
                stats["failed_documents"] = stats.get("failed_documents", 0) + 1
            
            self._save_manifest()

    def get_document_status(self, doc_path: Path) -> Optional[Dict]:
        """Get document processing status and metadata."""
        doc_id = str(doc_path)
        return self.manifest["documents"].get(doc_id)

    def get_changed_documents(self, docs_path: Path) -> List[Path]:
        """Identify documents that have changed or are new."""
        changed_docs = []
        for doc_path in docs_path.rglob("*"):
            if not doc_path.is_file():
                continue
                
            current_hash = self._compute_file_hash(doc_path)
            doc_id = str(doc_path)
            
            if doc_id not in self.manifest["documents"]:
                # New document
                changed_docs.append(doc_path)
            else:
                # Check if document has changed
                doc = self.manifest["documents"][doc_id]
                if doc["hash"] != current_hash:
                    changed_docs.append(doc_path)
                    
        return changed_docs

    def get_manifest_summary(self) -> Dict:
        """Get summary of manifest state."""
        return {
            "index_name": self.index_name,
            "total_documents": len(self.manifest["documents"]),
            "stats": self.manifest["stats"],
            "last_updated": self.manifest["last_updated"]
        }
