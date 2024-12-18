"""Node class for document tree."""
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import numpy as np
from src.storage.db_manager import DatabaseManager
from src.models.data_models import NodeData, EdgeData, TreeData
from uuid import uuid4
from collections import defaultdict, deque
import hashlib
import logging
import time
from queue import Queue

logger = logging.getLogger(__name__)

@dataclass
class LeafNode:
    """Document chunk node containing raw text."""
    id: str
    text: str                  # Raw document chunk
    level: int = 0            # Always 0 for leaves
    metadata: Dict = field(default_factory=dict)  # Technical metadata
    children: List = field(default_factory=list)  # Empty list for leaves
    
    def __hash__(self):
        """Make node hashable based on its ID."""
        return hash(self.id)
        
    def __eq__(self, other):
        """Nodes are equal if they have the same ID."""
        if not isinstance(other, LeafNode):
            return False
        return self.id == other.id
        
    def to_dict(self) -> Dict:
        """Convert node to dictionary format."""
        return {
            'id': self.id,
            'text': self.text,
            'level': self.level,
            'metadata': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.metadata.items()
            },
            'type': 'leaf'
        }

@dataclass 
class SummaryNode:
    """Internal node containing summarized content."""
    id: str
    text: str                  # Summarized content from children
    level: int                # 1+ (twig, branch, root)
    metadata: Dict = field(default_factory=dict)  # Node metadata
    children: List['Union[LeafNode, SummaryNode]'] = field(default_factory=list)
    
    def __hash__(self):
        """Make node hashable based on its ID."""
        return hash(self.id)
        
    def __eq__(self, other):
        """Nodes are equal if they have the same ID."""
        if not isinstance(other, SummaryNode):
            return False
        return self.id == other.id
        
    def add_child(self, child: Union['LeafNode', 'SummaryNode']) -> None:
        """Add a child node."""
        self.children.append(child)
        child.metadata['parent_id'] = self.id
        
    def to_dict(self) -> Dict:
        """Convert node to dictionary format."""
        return {
            'id': self.id,
            'text': self.text,
            'level': self.level,
            'metadata': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.metadata.items()
            },
            'type': 'summary',
            'children_ids': [child.id for child in self.children]
        }

@dataclass
class Node:
    """Node class for document tree.
    
    This class represents a node in the document tree and uses NodeData for its core data structure.
    It maintains parent-child relationships and provides methods for tree operations.
    
    Attributes:
        data (NodeData): Core node data using the NodeData class
        children (List[Node]): List of child nodes
        _parent (Optional[Node]): Parent node reference
        _db_manager (Optional[DatabaseManager]): Database manager instance
    """
    
    data: NodeData
    children: List['Node'] = field(default_factory=list)
    _parent: Optional['Node'] = None
    _db_manager: Optional[DatabaseManager] = None
    
    def __post_init__(self):
        """Initialize node after creation."""
        # Convert dict to NodeData if needed
        if isinstance(self.data, dict):
            self.data = NodeData(**self.data)
            
        # Initialize basic attributes if None
        if self.children is None:
            self.children = []
            
        # Ensure we have an ID
        if not self.data.id:
            self.data.id = str(uuid4())
            
        # Update metadata
        timestamp = time.time()
        self.data.metadata.setdefault('creation_time', float(timestamp))
        self.data.metadata.setdefault('last_modified', float(timestamp))
        self.data.metadata.setdefault('level', self.level)
        
        # Set up parent-child relationship if parent was provided
        if self._parent:
            self.parent = self._parent
            
    @property
    def parent(self) -> Optional['Node']:
        """Get parent node."""
        return self._parent
        
    @parent.setter 
    def parent(self, node: Optional['Node']) -> None:
        """Set parent node and update relationships."""
        # Check if tree is locked
        if hasattr(self, '_tree_locked') and self._tree_locked:
            raise RuntimeError("Cannot modify parent after tree is locked")
        
        # Prevent cycles by checking if the new parent is already a descendant
        def is_descendant(potential_child: 'Node', potential_parent: 'Node') -> bool:
            if not potential_child:
                return False
            if potential_child == potential_parent:
                return True
            return is_descendant(potential_child.parent, potential_parent)
            
        if node and is_descendant(node, self):
            raise ValueError(f"Cannot set node {node.id} as parent of {self.id} as it would create a cycle")
            
        # Remove from old parent's children if exists
        if self._parent and self in self._parent.children:
            self._parent.children.remove(self)
            self._parent.data.remove_child(self.id)
            
        # Update parent reference
        self._parent = node
        
        # Update NodeData parent_id
        if node:
            self.data.set_parent(node.id)
            # Add to new parent's children if not already there
            if self not in node.children:
                node.children.append(self)
                node.data.add_child(self.id)
        else:
            self.data.set_parent(None)
            
    def add_child(self, child: 'Node') -> None:
        """Add a child node."""
        # Check if tree is locked
        if hasattr(self, '_tree_locked') and self._tree_locked:
            raise RuntimeError("Cannot add children after tree is locked")
        
        if child not in self.children:
            self.children.append(child)
            self.data.add_child(child.id)
            child.parent = self
            
    def remove_child(self, child: 'Node') -> None:
        """Remove a child node while maintaining proper relationships.
        
        After removal, if this node has no more children, it becomes a leaf.
        """
        # Check if tree is locked
        if hasattr(self, '_tree_locked') and self._tree_locked:
            raise RuntimeError("Cannot remove children after tree is locked")
        
        if child in self.children:
            self.children.remove(child)
            child._parent = None
            child.data['parent_id'] = None
            child.metadata['parent_id'] = None
            
            # If we have no more children, we become a leaf
            if not self.children:
                self.metadata['node_type'] = 'leaf'
                self.metadata['is_leaf'] = True
            self.metadata['last_modified'] = int(time.time())
            self._validate_node()
            
            # Save changes to database
            if self._db_manager:
                self.save_to_db()
                child.save_to_db()

    @property
    def text(self) -> str:
        """Get node text."""
        return self.data.text
        
    @text.setter
    def text(self, value: str) -> None:
        """Set node text."""
        self.data.text = value
        
    @property
    def id(self) -> str:
        """Get node ID."""
        return self.data.id
        
    @id.setter 
    def id(self, value: str) -> None:
        """Set node ID."""
        self.data.id = value

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get node metadata."""
        return self.data.metadata
        
    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        """Set node metadata."""
        self.data.metadata = value
        
    @property
    def level(self) -> int:
        """Get node level."""
        return self.data.level
        
    @level.setter
    def level(self, value: int) -> None:
        """Set node level."""
        self.data.level = value
        
    @property
    def main_topic(self) -> Optional[str]:
        """Get main topic."""
        return self.data.metadata.get('main_topic')
        
    @main_topic.setter
    def main_topic(self, value: Optional[str]) -> None:
        """Set main topic."""
        self.data.metadata['main_topic'] = value
        
    @property
    def key_concepts(self) -> List[str]:
        """Get key concepts."""
        return self.data.metadata.get('key_concepts', [])
        
    @key_concepts.setter
    def key_concepts(self, value: List[str]) -> None:
        """Set key concepts."""
        self.data.metadata['key_concepts'] = value
        
    @property
    def dependencies(self) -> List[str]:
        """Get dependencies."""
        return self.data.metadata.get('dependencies', [])
        
    @dependencies.setter
    def dependencies(self, value: List[str]) -> None:
        """Set dependencies."""
        self.data.metadata['dependencies'] = value
        
    @property
    def relationships(self) -> List[Dict[str, Any]]:
        """Get relationships."""
        return self.data.metadata.get('relationships', [])
        
    @relationships.setter
    def relationships(self, value: List[Dict[str, Any]]) -> None:
        """Set relationships."""
        self.data.metadata['relationships'] = value
        
    @property
    def topics(self) -> List[str]:
        """Get topics."""
        return self.data.metadata.get('topics', [])
        
    @topics.setter
    def topics(self, value: List[str]) -> None:
        """Set topics."""
        self.data.metadata['topics'] = value
        
    @property
    def summary_metadata(self) -> Dict[str, Any]:
        """Get summary metadata."""
        return self.data.metadata.get('summary_metadata', {})
        
    @summary_metadata.setter
    def summary_metadata(self, value: Dict[str, Any]) -> None:
        """Set summary metadata."""
        self.data.metadata['summary_metadata'] = value
        
    @property
    def texts(self) -> List[str]:
        """Get node texts."""
        # For backward compatibility, if 'texts' exists in data, use it
        if 'texts' in self.data.metadata:
            return self.data.metadata['texts']
        # Otherwise, return a list containing just the text
        return [self.text] if self.text else []
        
    @texts.setter
    def texts(self, value: List[str]) -> None:
        """Set node texts."""
        self.data.metadata['texts'] = value
        
    @property
    def cluster_info(self) -> Dict[str, Any]:
        """Get cluster information."""
        return self.data.cluster_info
        
    @cluster_info.setter
    def cluster_info(self, value: Dict[str, Any]) -> None:
        """Set cluster information."""
        self.data.cluster_info = value
        
    def set_depth(self, depth: int) -> None:
        """Set the depth of this node."""
        self.depth = depth
        
    def to_node_data(self) -> NodeData:
        """Convert to standard NodeData format.
        
        Returns:
            NodeData object
        """
        return self.data
        
    def to_tree_data(self) -> TreeData:
        """Convert node and its subtree to standard TreeData format.
        
        Returns:
            TreeData object containing this node and all descendants
        """
        nodes = []
        edges = []
        
        def process_node(node: Node, parent_id: Optional[str] = None):
            node_data = node.to_node_data()
            nodes.append(node_data)
            
            if parent_id is not None:
                edges.append(EdgeData(
                    source=parent_id,
                    target=node_data.id
                ))
                
            for child in node.children:
                process_node(child, node_data.id)
                
        process_node(self)
        return TreeData(nodes=nodes, edges=edges)

    def update_embedding(self, embedding: np.ndarray) -> None:
        """Update node embedding and timestamp."""
        self.data.set_embedding(embedding)
        self.data.metadata['last_modified'] = int(time.time())

    def update_text(self, text: str) -> None:
        """Update node text and related metadata."""
        self.texts = [text]
        self.metadata.update({
            'last_modified': int(time.time())
        })

    def is_balanced(self, tolerance: float = 0.5) -> bool:
        """Check if the subtree rooted at this node is balanced.
        
        A tree is considered balanced if:
        1. The heights of any two sibling subtrees differ by at most 1 level
        2. The number of children in any two sibling nodes differs by at most 
           tolerance * max(number of children)
        3. Each non-leaf node has at least 2 children (except the root)
        """
        # If this is not the root and has only 1 child, it's unbalanced
        if self._parent and len(self.children) == 1:
            return False

        if not self.children:
            return True

        # Check height balance
        heights = [child.get_height() for child in self.children]
        if max(heights) - min(heights) > 1:
            return False

        # Check children count balance
        child_counts = [len(child.children) for child in self.children]
        if not child_counts:
            return True
            
        max_count = max(child_counts)
        if max_count == 0:
            return True
            
        max_diff = max_count * tolerance
        if max(child_counts) - min(child_counts) > max_diff:
            return False

        # Recursively check children
        return all(child.is_balanced(tolerance) for child in self.children)

    def count_nodes(self) -> int:
        """Count the total number of nodes in the tree."""
        count = 1  # Count this node
        for child in self.children:
            count += child.count_nodes()
        return count
        
    def get_max_depth(self) -> int:
        """Get the maximum depth of the tree."""
        if not self.children:
            return 0
        return 1 + max(child.get_max_depth() for child in self.children)

    def _convert_numpy_to_list(self, obj):
        """Convert numpy arrays to lists recursively."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        return obj

    def to_dict(self) -> Dict:
        """Convert node to dictionary representation."""
        data = {
            'id': self.id,
            'text': self.texts[0] if self.texts else '',
            'level': self.level,
            'metadata': {}
        }
        
        # Convert metadata, handling numpy arrays
        for key, value in self.data.metadata.items():
            if isinstance(value, np.ndarray):
                data['metadata'][key] = value.tolist()
            elif isinstance(value, (list, dict)):
                # Deep copy to avoid modifying original
                import copy
                data['metadata'][key] = copy.deepcopy(value)
            else:
                data['metadata'][key] = value
                
        # Add embedding if present
        embedding = self.data.get_embedding()
        if embedding is not None:
            data['metadata']['embedding'] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            data['metadata']['embedding_id'] = self.data.get_embedding_id()
                
        return data

    def to_tree_format(self) -> Dict:
        """Convert the entire tree to the format expected by the visualizer."""
        node_dict = {
            'id': self.id,
            'text': self.texts[0] if self.texts else '',
            'level': self.level,
            'parent_id': self.parent.id if self.parent else None,
            'children_ids': [child.id for child in self.children],
            'metadata': {}
        }
        
        # Convert metadata, handling numpy arrays
        for key, value in self.data.metadata.items():
            if isinstance(value, np.ndarray):
                node_dict['metadata'][key] = value.tolist()
            elif isinstance(value, (list, dict)):
                # Deep copy to avoid modifying original
                import copy
                node_dict['metadata'][key] = copy.deepcopy(value)
            else:
                node_dict['metadata'][key] = value
                
        # Add embedding if present
        embedding = self.data.get_embedding()
        if embedding is not None:
            node_dict['metadata']['embedding'] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            node_dict['metadata']['embedding_id'] = self.data.get_embedding_id()
                
        return node_dict

    def update_cluster_info(self, cluster_data: Dict) -> None:
        """Update node's cluster information based on summarization output"""
        if isinstance(cluster_data, dict):
            self.cluster_info["cluster_name"] = cluster_data.get("name")
            self.cluster_info["size"] = cluster_data.get("size", 1)
            if "key_concepts" in cluster_data:
                self.cluster_info["key_concepts"] = [
                    {
                        "concept": concept["concept"],
                        "importance": concept["importance_score"]
                    }
                    for concept in cluster_data["key_concepts"]
                ]
            if "relationships" in cluster_data:
                self.cluster_info["relationships"] = cluster_data["relationships"]

    def get_cluster_weight(self) -> float:
        """Calculate cluster weight based on size and importance scores"""
        base_weight = self.cluster_info["size"]
        importance_factor = sum(concept["importance"] for concept in self.cluster_info["key_concepts"]) / len(self.cluster_info["key_concepts"]) if self.cluster_info["key_concepts"] else 1
        relationship_factor = sum(rel["strength"] for rel in self.cluster_info["relationships"]) / len(self.cluster_info["relationships"]) if self.cluster_info["relationships"] else 1
        return base_weight * (importance_factor + relationship_factor) / 2

    def to_network_dict(self) -> Dict:
        """Convert tree to network visualization format with enhanced cluster information"""
        nodes = []
        edges = []
        queue = deque([(self, None)])  # (node, parent_id)
        
        while queue:
            node, parent_id = queue.popleft()
            
            # Create node data
            node_data = {
                'id': node.id,
                'label': node.cluster_info["cluster_name"] or "Unnamed Node",
                'size': node.get_cluster_weight(),
                'color': node._get_cluster_color(),
                'title': f"Size: {node.cluster_info['size']}<br>Weight: {node.get_cluster_weight():.2f}",
                'key_concepts': [concept["concept"] for concept in node.cluster_info["key_concepts"]],
                'group': node.cluster_info["cluster_name"]
            }
            nodes.append(node_data)
            
            # Create edge if not root
            if parent_id:
                edges.append({
                    'from': parent_id,
                    'to': node.id,
                    'value': max(1, min(node.get_cluster_weight() / 2, 10))  # Scale edge weight
                })
            
            # Add children to queue
            for child in node.children:
                queue.append((child, node.id))
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'max_level': max(n['level'] for n in nodes),
                'root_id': self.id
            }
        }

    def _get_cluster_color(self) -> str:
        """Generate consistent color based on cluster name for visualization"""
        if not self.cluster_info["cluster_name"]:
            return "#808080"  # Default gray for unnamed clusters
        # Generate consistent color hash from cluster name
        hash_value = hashlib.md5(self.cluster_info["cluster_name"].encode()).hexdigest()
        return f"#{hash_value[:6]}"

    def get_ancestor_at_level(self, target_level: int) -> Optional['Node']:
        """Get the ancestor node at the specified level."""
        if target_level > self.level:
            return None
        if target_level == self.level:
            return self
        
        current = self
        while current and current.level > target_level:
            current = current._parent
        return current
    
    def get_descendants(self, max_depth: int = None) -> List['Node']:
        """Get all descendant nodes up to max_depth."""
        descendants = []
        if max_depth is not None and max_depth < 0:
            return descendants
            
        for child in self.children:
            descendants.append(child)
            if max_depth is None or max_depth > 0:
                descendants.extend(child.get_descendants(
                    max_depth - 1 if max_depth is not None else None
                ))
        return descendants
    
    def get_leaves(self) -> List['Node']:
        """Get all leaf nodes in the subtree rooted at this node.
        
        Returns:
            List of leaf nodes
        """
        leaves = []
        if not self.children:  # If this is a leaf node
            leaves.append(self)
        else:
            for child in self.children:
                leaves.extend(child.get_leaves())
        return leaves
    
    def get_leaf_nodes(self) -> List['Node']:
        """Get all leaf nodes in this subtree."""
        if not self.children:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaf_nodes())
        return leaves
    
    def get_breadth(self) -> int:
        """Get the maximum breadth (number of nodes at any level) of this subtree."""
        queue = [(self, 0)]
        level_counts = defaultdict(int)
        level_counts[0] = 1
        
        while queue:
            node, level = queue.pop(0)
            for child in node.children:
                next_level = level + 1
                level_counts[next_level] += 1
                queue.append((child, next_level))
        
        return max(level_counts.values())

    def get_node_type_for_level(self) -> str:
        """Get node type based on level and position in tree.
        
        Node types follow a strict hierarchy:
        1. Root (level 0): Single node at top of tree
        2. Branch (level 1): Major document sections
        3. Twig (level 2): Document groups
        4. Leaf (level 3 or is_document_chunk): Individual document chunks
        
        Returns:
            str: One of 'root', 'branch', 'twig', 'leaf'
        """
        # Document chunks are ALWAYS leaves, regardless of level
        if self.metadata.get('is_document_chunk', False):
            self.metadata['node_type'] = 'leaf'
            return 'leaf'
        
        # Otherwise, type is determined strictly by level
        level_types = {
            0: 'root',
            1: 'branch',
            2: 'twig',
            3: 'leaf'
        }
        
        # Get node type based on level
        node_type = level_types.get(self.level, 'leaf')  # Default to leaf for unknown levels
        
        # Update metadata to maintain consistency
        self.metadata['node_type'] = node_type
        self.metadata['is_leaf'] = node_type == 'leaf'
        
        return node_type
        
    def validate_node_type(self) -> bool:
        """Validate that the node type matches its level and position in tree.
        
        Returns:
            bool: True if node type is valid, False otherwise
        """
        # Get expected type
        expected_type = self.get_node_type_for_level()
        
        # Check if current type matches expected
        current_type = self.metadata.get('node_type')
        if current_type != expected_type:
            logger.warning(
                f"Node type mismatch - expected {expected_type}, got {current_type}. "
                f"Level: {self.level}, is_document_chunk: {self.metadata.get('is_document_chunk', False)}"
            )
            return False
            
        # Validate that leaf nodes have no children
        if expected_type == 'leaf' and self.children:
            logger.warning(f"Leaf node {self.id} has children")
            return False
            
        # Validate that non-leaf nodes have children (except during tree construction)
        if expected_type != 'leaf' and not self.children and self.metadata.get('tree_construction_complete', False):
            logger.warning(f"Non-leaf node {self.id} has no children")
            return False
            
        return True
        
    def repair_node_type(self) -> None:
        """Repair node type to match its level and position in tree."""
        # Get correct type
        correct_type = self.get_node_type_for_level()
        
        # Update metadata
        self.metadata['node_type'] = correct_type
        self.metadata['is_leaf'] = correct_type == 'leaf'
        
        # Log the change
        logger.info(f"Repaired node {self.id} type to {correct_type}")
        
    @classmethod
    def from_dict(cls, data: Dict, parent: Optional['Node'] = None, _db_manager=None) -> 'Node':
        """Create a node from the standardized dictionary format.
        
        This is the canonical deserialization method for nodes across the codebase.
        All modules should use this method when deserializing nodes.
        """
        # Create node with basic attributes
        node = cls(
            data=data,
            parent=parent,
            _db_manager=_db_manager
        )
        
        # Set ID if provided, otherwise keep generated ID
        if 'id' in data:
            node.id = data['id']
            
        # Set metadata and cluster info
        node.metadata = data.get('metadata', {})
        node.cluster_info = data.get('cluster_info', {
            "cluster_name": None,
            "importance_score": 0.0,
            "size": 1,
            "relationships": [],
            "key_concepts": []
        })
        
        # Set embedding if present
        if 'embedding' in data:
            node.data.set_embedding(data['embedding'])
            
        return node

    def save_to_db(self) -> bool:
        """Save this node to the database."""
        if not self._db_manager:
            logger.warning("No database manager available for node")
            return False
            
        # Ensure document tracking metadata is included
        if 'doc_id' not in self.metadata:
            self.metadata.update({
                'doc_id': None,
                'doc_name': None,
                'chunk_num': None,
                'start_idx': None,
                'end_idx': None,
                'token_count': None
            })
                
        # Save node
        success = self._db_manager.save_node(
            node_id=self.id,
            parent_id=self._parent.id if self._parent else None,
            level=self.level,
            text=self.texts[0] if self.texts else '',
            metadata=self.metadata,
            embedding_id=self.data.get_embedding_id()
        )
        
        if not success:
            logger.error(f"Failed to save node {self.id} to database")
            return False
            
        # Save children recursively
        for child in self.children:
            child._db_manager = self._db_manager  # Ensure child has db_manager
            if not child.save_to_db():
                logger.error(f"Failed to save child node {child.id} to database")
                return False
                
        return True

    @classmethod
    def from_db(cls, db_manager: DatabaseManager, node_id: str) -> Optional['Node']:
        """Create a Node instance from database data."""
        # Get node data from database
        node_data = db_manager.get_node(node_id)
        if not node_data:
            return None
            
        # Create node instance with NodeData
        node = cls(
            data=node_data,
            _db_manager=db_manager
        )
        
        # Set ID if provided, otherwise keep generated ID
        if 'id' in node_data:
            node.id = node_data['id']
            
        # Load embedding from database
        embedding_id = node.data.get_embedding_id()
        if embedding_id:
            node.data.set_embedding(db_manager.get_embedding(embedding_id))
            
        # Load children recursively
        children_ids = db_manager.get_children_ids(node_id)
        for child_id in children_ids:
            child = cls.from_db(db_manager, child_id)
            if child:
                child.parent = node
                node.children.append(child)
                
        return node

    def __hash__(self) -> int:
        """Make Node hashable based on its ID."""
        return hash(self.id)
        
    def __eq__(self, other) -> bool:
        """Nodes are equal if they have the same ID."""
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def remove_child(self, child: 'Node') -> None:
        """Remove a child node while maintaining proper relationships.
        
        After removal, if this node has no more children, it becomes a leaf.
        """
        # Check if tree is locked
        if hasattr(self, '_tree_locked') and self._tree_locked:
            raise RuntimeError("Cannot remove children after tree is locked")
        
        if child in self.children:
            self.children.remove(child)
            child._parent = None
            child.data['parent_id'] = None
            child.metadata['parent_id'] = None
            
            # If we have no more children, we become a leaf
            if not self.children:
                self.metadata['node_type'] = 'leaf'
                self.metadata['is_leaf'] = True
            self.metadata['last_modified'] = int(time.time())
            self._validate_node()
            
            # Save changes to database
            if self._db_manager:
                self.save_to_db()
                child.save_to_db()

    def get_siblings(self) -> List['Node']:
        """Get all sibling nodes."""
        if not self._parent:
            return []
        return [node for node in self._parent.children if node != self]

    def get_depth(self) -> int:
        """Calculate the depth of this node (distance from root)."""
        depth = 0
        current = self
        while current._parent:
            depth += 1
            current = current._parent
        return depth

    def get_height(self) -> int:
        """Calculate the height of this node (length of longest path to leaf).
        
        The height of a leaf node is 0, and each level above adds 1.
        For example:
        - A single node (leaf) has height 0
        - A root with one child has height 1
        - A root with a child that has a child (depth 2) has height 2
        """
        if not self.children:
            return 0  # Height of a leaf node is 0
        return 1 + max(child.get_height() for child in self.children)
