"""Data models for tree structures.

This module contains the core data models used to represent tree structures,
including nodes, edges, and the tree itself. These models provide a standardized
way to store and manipulate tree data across the system.

The data models follow these principles:
1. Each node has a unique ID and maintains proper parent-child relationships
2. Nodes are organized hierarchically with well-defined levels
3. All node metadata follows a consistent format
4. Edges represent connections between nodes with optional weights and metadata

Classes:
    NodeData: Data class for node information
    EdgeData: Data class for edge information
    TreeData: Data class for entire tree structure
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import uuid
import numpy as np

@dataclass
class NodeData:
    """Data class for node information.
    
    This class contains all information about a single node, including its content,
    position in the tree, and any additional metadata.
    
    Attributes:
        id (str): Unique identifier for the node. Should be consistent across the system.
        text (str): The actual text content of the node.
        level (int): The depth level in the tree (0 for root).
        parent_id (str): ID of the parent node (None for root).
        children_ids (List[str]): List of child node IDs.
        metadata (Dict[str, Any]): Core node information, which includes:
            - creation_time (float): Node creation timestamp
            - last_modified (float): Last modification timestamp
            - is_leaf (bool): Whether this is a leaf node
            - node_type (str): Type of node (root, branch, leaf)
            - title (str): Short title or summary
            - cluster_score (float): Node's clustering score (0-1)
            - importance_score (float): Node's importance in the document (0-1)
            - summary (str): Generated summary of node content
            - main_topics (List[str]): Main topics identified in the node
            - embedding (np.ndarray): Node's embedding vector
            - embedding_id (str): ID of stored embedding if applicable
        document_info (Dict[str, Any]): Document processing information including:
            - filename (str): Name of the source file
            - path (str): Full path to the source file
            - size (int): File size in bytes
            - modified (float): File modification timestamp
            - type (str): File type/extension
            - chunk_index (int): Index of this chunk in the document
            - total_chunks (int): Total number of chunks in the document
            - token_count (int): Number of tokens in this chunk
            - char_count (int): Number of characters in this chunk
            - is_first_chunk (bool): Whether this is the first chunk
            - is_final_chunk (bool): Whether this is the final chunk
            - embedding_model (str): Model used for embeddings
            - embedding_time (float): Time taken to generate embeddings
            - embedding_batch (int): Batch number for embedding generation
            - embedding_position (int): Position in the embedding batch
            - is_document_chunk (bool): Whether this is a document chunk
        leaf (Dict[str, Any]): Information about leaf nodes containing:
            - filename (str): Real name of the file that the chunk belongs to
            - total_chunks (int): Total number of chunks the file was split into
            - chunk_index (int): Consecutive number of the chunk (1-based indexing)
        key_concepts (List[Dict]): Key concepts with structure:
            - concept (str): Name of concept
            - description (str): Brief description
            - importance_score (str): Score from 1-10
            - cluster_relevance (str): How concept relates to cluster
        technical_details (Dict): Technical information including:
            - implementation (List[str]): Key implementation details
            - algorithms (List[str]): Key algorithms used
            - data_structures (List[str]): Important data structures
            - dependencies (List[str]): External dependencies
            - cluster_patterns (List[str]): Common patterns in cluster
        relationships (List[Dict]): Relationships with structure:
            - from (str): Source component
            - to (str): Target component
            - type (str): Relationship type
            - description (str): Brief description
            - cluster_context (str): How relationship affects cluster
        cluster_insights (Dict): Cluster-specific information:
            - unifying_theme (str): What makes items belong together
            - distinguishing_features (List[str]): What makes cluster unique
            - cluster_patterns (List[str]): Common patterns in cluster
            - sub_clusters (List[Dict]): Insights from child clusters
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    level: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_info: Dict[str, Any] = field(default_factory=lambda: {
        "filename": "",
        "path": "",
        "size": 0,
        "modified": 0.0,
        "type": "",
        "chunk_index": 0,
        "total_chunks": 0,
        "token_count": 0,
        "char_count": 0,
        "is_first_chunk": False,
        "is_final_chunk": False,
        "embedding_model": "",
        "embedding_time": 0.0,
        "embedding_batch": 0,
        "embedding_position": 0,
        "is_document_chunk": False
    })
    leaf: Dict[str, Any] = field(default_factory=lambda: {"filename": "", "total_chunks": 0, "chunk_index": 0})
    key_concepts: List[Dict] = field(default_factory=list)
    technical_details: Dict = field(default_factory=lambda: {
        "implementation": [],
        "algorithms": [],
        "data_structures": [],
        "dependencies": [],
        "cluster_patterns": []
    })
    relationships: List[Dict] = field(default_factory=list)
    cluster_insights: Dict = field(default_factory=lambda: {
        "unifying_theme": "",
        "distinguishing_features": [],
        "cluster_patterns": [],
        "sub_clusters": []
    })
    
    def __post_init__(self):
        """Initialize metadata after creation."""
        if not self.metadata:
            self.metadata = {}
            
        # Ensure required metadata fields exist
        defaults = {
            'creation_time': time.time(),
            'last_modified': time.time(),
            'is_leaf': False,
            'node_type': 'leaf' if self.metadata.get('is_leaf', False) else 'branch',
            'title': '',
            'cluster_score': 0.0,
            'importance_score': 0.0,
            'summary': '',
            'main_topics': [],
            'embedding': None,
            'embedding_id': ''
        }
        
        for key, value in defaults.items():
            if key not in self.metadata:
                self.metadata[key] = value
                
        # Ensure parent_id is in metadata for backward compatibility
        if self.parent_id is not None:
            self.metadata['parent_id'] = self.parent_id
            
    def get_embedding(self) -> Optional[np.ndarray]:
        """Get node's embedding vector.
        
        Returns:
            np.ndarray if embedding exists, None otherwise
        """
        embedding = self.metadata.get('embedding')
        if embedding is None:
            return None
            
        # Convert list to numpy array if needed
        if isinstance(embedding, list):
            return np.array(embedding)
        elif isinstance(embedding, np.ndarray):
            return embedding
        else:
            return None
            
    def get_embedding_id(self) -> str:
        """Get node's embedding ID.
        
        Returns:
            str: The embedding ID if it exists, empty string otherwise
        """
        return self.metadata.get('embedding_id', '')
            
    def set_embedding(self, embedding: Optional[np.ndarray]):
        """Set node's embedding vector.
        
        Args:
            embedding: Numpy array or None
        """
        if embedding is None:
            self.metadata['embedding'] = None
            self.metadata['embedding_id'] = ''
        else:
            # Convert numpy array to list for JSON serialization
            self.metadata['embedding'] = embedding.tolist()
            if not self.metadata.get('embedding_id'):
                self.metadata['embedding_id'] = str(uuid.uuid4())
                
    def add_child(self, child_id: str):
        """Add a child node ID.
        
        Args:
            child_id: ID of the child node to add
        """
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
            self.metadata['last_modified'] = time.time()
            
    def remove_child(self, child_id: str):
        """Remove a child node ID.
        
        Args:
            child_id: ID of the child node to remove
        """
        if child_id in self.children_ids:
            self.children_ids.remove(child_id)
            self.metadata['last_modified'] = time.time()
            
    def set_parent(self, parent_id: Optional[str]):
        """Set the parent node ID.
        
        Args:
            parent_id: ID of the parent node (None for root)
        """
        self.parent_id = parent_id
        if parent_id is not None:
            self.metadata['parent_id'] = parent_id
        elif 'parent_id' in self.metadata:
            del self.metadata['parent_id']
        self.metadata['last_modified'] = time.time()
        
    def update_metadata(self, **kwargs):
        """Update node metadata.
        
        Args:
            **kwargs: Key-value pairs to update in metadata
        """
        self.metadata.update(kwargs)
        self.metadata['last_modified'] = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert node data to dictionary format.
        
        Returns:
            Dict containing serialized node data
        """
        # Convert numpy arrays to lists in metadata
        metadata = {}
        for key, value in self.metadata.items():
            if isinstance(value, np.ndarray):
                metadata[key] = value.tolist()
            elif isinstance(value, list):
                metadata[key] = [
                    item.tolist() if isinstance(item, np.ndarray) else item 
                    for item in value
                ]
            elif isinstance(value, dict):
                # Handle nested dictionaries
                nested_dict = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        nested_dict[k] = v.tolist()
                    elif isinstance(v, list):
                        nested_dict[k] = [
                            i.tolist() if isinstance(i, np.ndarray) else i 
                            for i in v
                        ]
                    else:
                        nested_dict[k] = v
                metadata[key] = nested_dict
            else:
                metadata[key] = value

        return {
            'id': self.id,
            'text': self.text,
            'level': self.level,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'metadata': metadata,
            'document_info': self.document_info,
            'leaf': self.leaf,
            'key_concepts': self.key_concepts,
            'technical_details': self.technical_details,
            'relationships': self.relationships,
            'cluster_insights': self.cluster_insights
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeData':
        """Create NodeData from dictionary.
        
        Args:
            data: Dictionary containing node data
            
        Returns:
            NodeData instance
        """
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            text=data.get('text', ''),
            level=data.get('level', 0),
            parent_id=data.get('parent_id'),
            children_ids=data.get('children_ids', []),
            metadata=data.get('metadata', {}),
            document_info=data.get('document_info', {
                "filename": "",
                "path": "",
                "size": 0,
                "modified": 0.0,
                "type": "",
                "chunk_index": 0,
                "total_chunks": 0,
                "token_count": 0,
                "char_count": 0,
                "is_first_chunk": False,
                "is_final_chunk": False,
                "embedding_model": "",
                "embedding_time": 0.0,
                "embedding_batch": 0,
                "embedding_position": 0,
                "is_document_chunk": False
            }),
            leaf=data.get('leaf', {"filename": "", "total_chunks": 0, "chunk_index": 0}),
            key_concepts=data.get('key_concepts', []),
            technical_details=data.get('technical_details', {
                "implementation": [],
                "algorithms": [],
                "data_structures": [],
                "dependencies": [],
                "cluster_patterns": []
            }),
            relationships=data.get('relationships', []),
            cluster_insights=data.get('cluster_insights', {
                "unifying_theme": "",
                "distinguishing_features": [],
                "cluster_patterns": [],
                "sub_clusters": []
            })
        )

@dataclass
class EdgeData:
    """Data class for edge information.
    
    Edges define the relationships between nodes, including the strength and nature
    of the connection.
    
    Attributes:
        source (str): ID of the source node
        target (str): ID of the target node
        weight (float): Connection strength (0-1), where:
            - 1.0: Strong direct connection
            - 0.7-0.9: Strong topical relationship
            - 0.4-0.6: Moderate relationship
            - 0.1-0.3: Weak or indirect relationship
        metadata (Dict[str, Any]): Additional edge information, which may include:
            - creation_time (float): Edge creation timestamp
            - last_modified (float): Last modification timestamp
            - relationship_type (str): Type of relationship
            - similarity_score (float): Semantic similarity between nodes
            - connection_strength (float): Alternative measure of connection
    """
    source: str
    target: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize metadata after creation."""
        timestamp = time.time()
        default_metadata = {
            'creation_time': timestamp,
            'last_modified': timestamp,
            'relationship_type': 'parent-child'
        }
        # Update metadata with defaults while preserving any existing values
        self.metadata = {**default_metadata, **self.metadata}
    
    def update_metadata(self, **kwargs):
        """Update edge metadata.
        
        Args:
            **kwargs: Key-value pairs to update in metadata
        """
        self.metadata.update(kwargs)
        self.metadata['last_modified'] = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge data to dictionary format.
        
        Returns:
            Dict containing serialized edge data
        """
        return {
            'source': self.source,
            'target': self.target,
            'weight': self.weight,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EdgeData':
        """Create EdgeData from dictionary.
        
        Args:
            data: Dictionary containing edge data
            
        Returns:
            EdgeData instance
        """
        return cls(
            source=data['source'],
            target=data['target'],
            weight=data.get('weight', 1.0),
            metadata=data.get('metadata', {})
        )

@dataclass
class TreeData:
    """Data class for tree structure.
    
    This class represents an entire tree structure, containing all nodes, edges,
    and associated metadata. It provides methods for adding and managing nodes
    and edges, as well as serialization capabilities.
    
    The tree structure follows these rules:
    1. Each node has a unique ID and maintains proper parent-child relationships
    2. Nodes are organized hierarchically with well-defined levels
    3. All node metadata follows the standard format defined in NodeData
    4. Tree operations preserve the integrity of node relationships and metadata
    
    Attributes:
        nodes (List[NodeData]): All nodes in the tree
        edges (List[EdgeData]): All edges connecting the nodes
        metadata (Dict[str, Any]): Tree-level metadata, which may include:
            - created_at (float): Tree creation timestamp
            - last_modified (float): Last modification timestamp
            - document_id (str): Original document identifier
            - version (str): Tree structure version
            - clustering_params (Dict): Parameters used for clustering
            - quality_metrics (Dict): Tree quality assessment metrics
            - stats (Dict): Tree statistics (depth, breadth, etc.)
            - source_info (Dict): Information about the source document
            - cluster_metadata (Dict): Cluster-specific information:
                - total_clusters (int): Total number of clusters
                - cluster_sizes (Dict): Distribution of cluster sizes
                - cluster_depths (Dict): Distribution of cluster depths
                - cluster_coherence (float): Overall cluster coherence score
                - inter_cluster_similarity (float): Similarity between clusters
                - cluster_hierarchy (Dict): Hierarchical cluster relationships
                - cluster_themes (List[Dict]): Major themes per cluster level:
                    - level (int): Tree level (0 for root)
                    - themes (List[str]): Common themes at this level
                    - patterns (List[str]): Common patterns at this level
            - summarization_params (Dict): Parameters used for summarization:
                - model (str): Model used for summarization
                - language (str): Target language
                - max_tokens (int): Token limit per summary
                - temperature (float): Model temperature setting
    """
    nodes: List[NodeData] = field(default_factory=list)
    edges: List[EdgeData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize tree metadata after creation."""
        timestamp = time.time()
        default_metadata = {
            'created_at': timestamp,
            'last_modified': timestamp,
            'version': '1.0.0',
            'stats': {
                'depth': self.calculate_depth(),
                'node_count': len(self.nodes),
                'edge_count': len(self.edges)
            }
        }
        # Update metadata with defaults while preserving any existing values
        self.metadata = {**default_metadata, **self.metadata}
    
    def add_node(self, node: NodeData):
        """Add node to tree data.
        
        Args:
            node: NodeData instance to add
        """
        self.nodes.append(node)
        self._update_stats()
    
    def add_edge(self, edge: EdgeData):
        """Add edge between nodes.
        
        Args:
            edge: EdgeData instance to add
            
        Raises:
            ValueError: If source or target node doesn't exist
        """
        # Verify that both nodes exist
        source_exists = any(n.id == edge.source for n in self.nodes)
        target_exists = any(n.id == edge.target for n in self.nodes)
        
        if not source_exists:
            raise ValueError(f"Source node {edge.source} not found in tree")
        if not target_exists:
            raise ValueError(f"Target node {edge.target} not found in tree")
        
        self.edges.append(edge)
        self._update_stats()
    
    def get_node_by_id(self, node_id: str) -> Optional[NodeData]:
        """Get node by its ID.
        
        Args:
            node_id: ID of the node to find
            
        Returns:
            NodeData if found, None otherwise
        """
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_children(self, node_id: str) -> List[NodeData]:
        """Get all children of a node.
        
        Args:
            node_id: ID of the parent node
            
        Returns:
            List of child nodes
        """
        return [
            self.get_node_by_id(edge.target)
            for edge in self.edges
            if edge.source == node_id and (node := self.get_node_by_id(edge.target)) is not None
        ]
    
    def get_parent(self, node_id: str) -> Optional[NodeData]:
        """Get parent of a node.
        
        Args:
            node_id: ID of the child node
            
        Returns:
            Parent node if found, None otherwise
        """
        for edge in self.edges:
            if edge.target == node_id:
                return self.get_node_by_id(edge.source)
        return None
    
    def calculate_depth(self) -> int:
        """Calculate maximum depth of the tree.
        
        Returns:
            Maximum depth (level) in the tree
        """
        return max((node.level for node in self.nodes), default=0)
    
    def _update_stats(self):
        """Update tree statistics in metadata."""
        self.metadata['last_modified'] = time.time()
        self.metadata['stats'] = {
            'depth': self.calculate_depth(),
            'node_count': len(self.nodes),
            'edge_count': len(self.edges)
        }
    
    def update_metadata(self, **kwargs):
        """Update tree metadata.
        
        Args:
            **kwargs: Key-value pairs to update in metadata
        """
        self.metadata.update(kwargs)
        self.metadata['last_modified'] = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree data to dictionary format.
        
        Returns:
            Dict containing serialized tree data
        """
        return {
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges],
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreeData':
        """Create TreeData from dictionary.
        
        Args:
            data: Dictionary containing tree data
            
        Returns:
            TreeData instance
        """
        tree = cls()
        
        # Create nodes first
        for node_data in data.get('nodes', []):
            node = NodeData.from_dict(node_data)
            tree.add_node(node)
            
        # Create edges
        for edge_data in data.get('edges', []):
            edge = EdgeData.from_dict(edge_data)
            tree.add_edge(edge)
            
        # Set metadata
        tree.metadata = data.get('metadata', {})
        
        return tree
    
    def validate(self) -> bool:
        """Validate tree structure.
        
        Checks:
        1. All nodes have unique IDs
        2. All edges reference existing nodes
        3. No cycles in the graph
        4. Single root node
        5. Consistent level assignments
        
        Returns:
            True if valid, raises ValueError otherwise
        
        Raises:
            ValueError: If any validation check fails
        """
        # Check unique node IDs
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Duplicate node IDs found")
            
        # Check edge references
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(f"Edge references non-existent source node: {edge.source}")
            if edge.target not in node_ids:
                raise ValueError(f"Edge references non-existent target node: {edge.target}")
                
        # Check for cycles
        visited = set()
        def has_cycle(node_id: str, path: set) -> bool:
            if node_id in path:
                return True
            if node_id in visited:
                return False
            visited.add(node_id)
            path.add(node_id)
            for edge in self.edges:
                if edge.source == node_id:
                    if has_cycle(edge.target, path):
                        return True
            path.remove(node_id)
            return False
            
        # Find root nodes by checking parent-child relationships
        root_nodes = []
        for node in self.nodes:
            is_child = False
            for edge in self.edges:
                if edge.target == node.id:
                    is_child = True
                    break
            if not is_child:
                root_nodes.append(node.id)
            
        if not root_nodes:
            raise ValueError("No root node found")
        if len(root_nodes) > 1:
            raise ValueError("Multiple root nodes found")
            
        if has_cycle(root_nodes[0], set()):
            raise ValueError("Cycle detected in tree")
            
        # Check level consistency
        for edge in self.edges:
            source_node = self.get_node_by_id(edge.source)
            target_node = self.get_node_by_id(edge.target)
            if source_node and target_node:
                if target_node.level >= source_node.level:
                    raise ValueError(
                        f"Invalid level assignment: {source_node.id}({source_node.level}) -> "
                        f"{target_node.id}({target_node.level}). Child level must be less than parent level."
                    )
        
        return True
