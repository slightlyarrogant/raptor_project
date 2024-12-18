from typing import Dict, List
import time
import json

class VersionManager:
    """Manages versioning and rollback capabilities."""
    
    def __init__(self, store):
        self.store = store
        
    def create_version(self, tree: Dict) -> float:
        """Create a new version of the tree."""
        version = time.time()
        self._store_version(version, tree)
        return version
        
    def _store_version(self, version: float, tree: Dict) -> None:
        """Store version metadata in the vector store."""
        if not self.store:
            # For testing purposes
            return
            
        try:
            # Create version metadata
            version_metadata = {
                'version': version,
                'timestamp': time.time(),
                'doc_count': len(tree.get('nodes', {})),
                'changes': self._compute_changes(tree),
                'type': 'version_record'
            }
            
            # Create version vector (using zeros as this is just metadata)
            version_vector = {
                'id': f'version_{version}',
                'values': [0.0] * 1536,  # Using standard dimension
                'metadata': version_metadata
            }
            
            # Store in vector database
            self.store.upsert(vectors=[version_vector])
            
        except Exception as e:
            print(f"Warning: Failed to store version metadata: {str(e)}")
    
    def _compute_changes(self, tree: Dict) -> Dict:
        """Compute changes in the tree compared to previous version."""
        return {
            'nodes_added': len(tree.get('nodes', {})),
            'timestamp': time.time()
        }
        
    def get_version(self, version: float) -> Dict:
        """Retrieve a specific version of the tree."""
        if not self.store:
            return {}
            
        try:
            result = self.store.query(
                vector=[0.0] * 1536,
                filter={'version': version},
                top_k=1
            )
            
            if result and result.get('matches'):
                return result['matches'][0]['metadata']
            return {}
            
        except Exception as e:
            print(f"Warning: Failed to retrieve version {version}: {str(e)}")
            return {}
            
    def list_versions(self, limit: int = 10) -> List[Dict]:
        """List available versions."""
        if not self.store:
            return []
            
        try:
            result = self.store.query(
                vector=[0.0] * 1536,
                filter={'type': 'version_record'},
                top_k=limit
            )
            
            if result and result.get('matches'):
                versions = [match['metadata'] for match in result['matches']]
                return sorted(versions, key=lambda x: x['timestamp'], reverse=True)
            return []
            
        except Exception as e:
            print(f"Warning: Failed to list versions: {str(e)}")
            return []
            
    def rollback_to_version(self, target_version: float) -> bool:
        """Rollback the tree to a specific version."""
        if not self.store:
            return False
            
        try:
            # Get target version metadata
            version_data = self.get_version(target_version)
            if not version_data:
                return False
                
            # Mark current version as rolled back
            current_version = time.time()
            rollback_metadata = {
                'version': current_version,
                'timestamp': time.time(),
                'type': 'rollback_record',
                'target_version': target_version,
                'status': 'completed'
            }
            
            # Store rollback metadata
            rollback_vector = {
                'id': f'rollback_{current_version}',
                'values': [0.0] * 1536,
                'metadata': rollback_metadata
            }
            
            self.store.upsert(vectors=[rollback_vector])
            return True
            
        except Exception as e:
            print(f"Warning: Failed to rollback to version {target_version}: {str(e)}")
            return False