# RAPTOR System Documentation

## Overview
RAPTOR (Recursive Analysis and Processing Tool for Organized Retrieval) is a document processing and analysis system designed for handling technical documentation with incremental knowledge integration capabilities.

## Key Components and Flow

### 1. Command Line Interface
The system can be initiated in several ways:

```bash
# Basic processing with analysis
python -m src.scripts.process_with_analysis --index-name raptor-cfi

# Incremental processing with watch mode
python -m src.scripts.process_with_analysis \
    --index-name your-index \
    --input-dir /path/to/documents \
    --watch-mode True \
    --batch-size 10

# Start visualization server
python -m src.visualization.server \
    --port 8080 \
    --metrics-interval 5
```

### 2. System Architecture

#### 2.1 Document Processing Pipeline
```
New Documents → Pre-processing → Delta Analysis → Tree Update → Vector Store Update
```

#### 2.2 Core Components
- Document Delta Detection (versioning using SHA-256 hashes)
- Tree Management (local restructuring, subtree rebuilding, full rebalancing)
- Vector Store Operations (namespace versioning, batch operations)
- Interactive Visualization Server
- Document Queue Manager
- System Monitoring Dashboard

### 3. Data Structures

#### 3.1 Document Manifest
```json
{
    "doc_id": {
        "path": "string",
        "hash": "string",
        "last_processed": "timestamp",
        "chunks": ["chunk_ids"],
        "tree_nodes": ["node_ids"],
        "metadata": {
            "version": "string",
            "processing_status": "string"
        }
    }
}
```

#### 3.2 Vector Metadata
```json
{
    "vector_id": {
        "version": "string",
        "source_doc": "string",
        "creation_date": "timestamp",
        "update_history": ["changes"],
        "tree_position": {
            "level": "int",
            "parent": "string",
            "children": ["strings"]
        }
    }
}
```

#### 3.3 Core Data Models

The RAPTOR system uses several key data models to ensure consistency and type safety throughout the codebase:

#### NodeData Structure
Nodes in the RAPTOR system have different structures based on their level:

##### Leaf Nodes (Level 0)
Document chunk nodes containing raw text:
```python
@dataclass
class LeafNodeData:
    id: str
    text: str                  # Raw document chunk (500-1000 tokens)
    level: int = 0            # Always 0 for leaves
    metadata: Dict = field(default_factory=dict)  # Technical metadata:
        # - filename: str        # Source file name
        # - path: str           # Source file path
        # - chunk_index: int    # Position in original document
        # - total_chunks: int   # Total chunks in document
        # - tokens: int         # Token count
        # - embedding: List[float] # Vector representation
        # - embedding_model: str # Model used for embedding
        # - creation_time: float # Node creation timestamp
```

##### Summary Nodes (Level 1+)
Internal nodes containing summarized content:
```python
@dataclass
class SummaryNodeData:
    id: str
    text: str                  # Summarized content from children
    level: int                # 1+ (twig, branch, root)
    metadata: Dict = field(default_factory=dict)  # Node metadata:
        # - node_type: str      # "twig", "branch", "root"
        # - main_topics: List   # Key topics from summarization
        # - cluster_score: float # Quality of clustering
        # - creation_time: float # Node creation timestamp
```

##### EdgeData
Represents relationships between nodes:
```python
@dataclass
class EdgeData:
    source_id: str             # ID of the source node
    target_id: str             # ID of the target node
    weight: float = 1.0        # Edge weight (default=1.0)
    metadata: Dict = field(default_factory=dict)  # Additional edge properties
```

##### TreeData
Represents the complete tree structure:
```python
@dataclass
class TreeData:
    nodes: List[NodeData]      # List of all nodes in the tree
    edges: List[EdgeData]      # List of all edges
    metadata: Dict = field(default_factory=dict)  # Tree-level metadata including:
        # - total_documents: int          # Total number of documents
        # - last_updated: datetime        # Last modification time
        # - version: str                  # Tree version identifier
```

Example tree structure (as shown in sample data):
```json
{
    "tree": {
        "nodes": [
            {
                "id": "root",
                "level": 0,                # Root level
                "texts": ["Root Document"],
                "title": "Document Collection"
            },
            {
                "id": "branch1",
                "level": 1,                # Branch level
                "parent_id": "root",
                "texts": ["Technical Documents"]
            },
            {
                "id": "twig1_1",
                "level": 2,                # Leaf level
                "parent_id": "branch1",
                "texts": ["API Documentation"]
            }
        ],
        "metadata": {
            "total_nodes": 3,
            "max_depth": 2
        }
    }
}
```

### 3.3 Tree Management and Locking

The RAPTOR system implements a robust tree locking mechanism to prevent concurrent modifications and ensure data integrity:

#### Tree Locking Mechanism
```python
# Lock file structure (stored as {index_name}_tree.lock)
{
    "is_locked": true,
    "lock_id": "unique-uuid",
    "lock_time": timestamp,
    "created_at": "YYYY-MM-DD HH:MM:SS"
}
```

#### Key Features
- **Persistent Locks**: Lock state persists between program runs using lock files
- **Unique Instance IDs**: Each tree instance gets a unique UUID
- **Automatic Locking**: Trees are automatically locked after successful processing
- **Lock Validation**: Prevents creation of new TreeManager instances if a locked tree exists

#### Usage Guidelines
1. **Creating New Trees**:
   - Ensure no locked tree exists before creating a new TreeManager instance
   - Trees are automatically locked after successful document processing
   
2. **Modifying Trees**:
   - Locked trees cannot be modified
   - Must explicitly unlock a tree before modifications
   - New processing attempts on locked trees will raise errors

3. **Lock Management**:
   - Use `_unlock_tree()` to release locks when needed
   - Check lock status through TreeManager instance
   - Lock files are automatically cleaned up when trees are unlocked

4. **Error Handling**:
   - Clear error messages indicate lock status and creation time
   - Includes lock ID for tracking different tree instances
   - Automatic validation prevents accidental tree overwrites

#### Best Practices
- Always check for existing locks before creating new tree instances
- Use proper error handling around tree operations
- Clean up lock files if processing fails
- Monitor lock status in long-running operations

### 4. Directory Structure
```
src/
├── analysis/      # Document analysis
├── clustering/    # Clustering logic
├── embedding/     # Embedding management
├── monitoring/    # System monitoring
├── prepare/      # Data preparation
├── scripts/      # Main scripts
├── storage/      # Storage management
├── summarization/ # Text summarization
├── tree/         # Tree management
├── update/       # Tree updates
├── utils/        # Utilities
└── visualization/ # Visualization server

data/
├── {index_name}/
│   ├── raw/        # Input documents
│   ├── processed/  # Successfully processed documents
│   ├── failed/     # Failed documents
│   └── archive/    # Archived documents

test_outputs/
├── tree_viz/
│   └── {index_name}/  # Visualization outputs
├── analysis/
│   └── {index_name}/  # Analysis results
└── reports/
    └── {index_name}/  # Analysis reports
```

### 5. Key Implementation Details

#### 5.1 Pinecone Integration
- Uses singleton pattern via pinecone_utils.py
- Index name from command line determines namespace
- Structure:
  ```
  namespace: {index_name}_namespace
  chunks: {index_name}_namespace_chunks
  tree: {index_name}_tree
  ```

#### 5.2 OpenAI Usage
- Embeddings: text-embedding-3-small
- Summarization: gpt-4-turbo-preview
- Fallback mechanisms for failures
- Token tracking and cost monitoring

#### 5.3 Tree Structure and Locking Mechanism

##### Tree Construction and Immutability
- Hierarchical organization with adaptive multi-level clustering
- Tree depth determined by log10 of document count:
  - 10 chunks -> 1 level (root[1] -> [chunks(0)])
  - 100 chunks -> 2 levels (root[2] -> branch[1] -> [chunks(0)])
  - 1000 chunks -> 3 levels (root[3] -> branch[2] -> twig[1] -> [chunks(0)])
  - Document chunks always at level 0
   
- Bottom-up tree construction:
  1. Create document chunk nodes at level 0 (leaves)
  2. Cluster level 0 nodes into level 1 (twigs)
  3. Cluster level 1 nodes into level 2 (branches)
  4. Create root node at highest level
  5. Back-propagate parent relationships
  6. Lock tree structure to prevent modifications
   
##### Tree Locking Mechanism
The tree implements a strict locking mechanism to ensure data integrity:

1. **Lock Timing**:
   - Tree is locked immediately after construction
   - Lock state is propagated to all nodes in the tree
   - Once locked, structural modifications are prohibited

2. **Protected Operations**:
   - Node addition/removal
   - Parent-child relationship changes
   - Level modifications
   - Structural metadata updates

3. **Lock Implementation**:
   - Each node maintains a `_tree_locked` flag
   - Lock state is inherited from parent nodes
   - All modification methods check lock state before execution
   - Attempts to modify locked trees raise RuntimeError

4. **Lock Propagation**:
   ```python
   def _propagate_lock(self, node: Node) -> None:
       node._tree_locked = True
       for child in node.children:
           self._propagate_lock(child)
   ```

5. **Modification Prevention**:
   - add_child(): Checks lock before adding new children
   - remove_child(): Checks lock before removing children
   - parent setter: Checks lock before changing relationships
   - metadata updates: Certain metadata fields locked after construction

##### Node Types and Rules
- Document chunks are always at level 0 (leaves)
- Each level has a specific role:
  - Level 0: Document chunks (actual content)
  - Level 1: Twig (subtopic grouping)
  - Level 2: Branch (topic clustering)
  - Level 3: Root (organizational)
- Higher levels are created as needed based on document count

##### Important Implementation Notes
1. **Tree Modification**:
   - All structural changes must occur during tree construction
   - Use TreeUpdater for controlled modifications when needed
   - Consider tree rebuilding for major structural changes

2. **Lock Verification**:
   ```python
   def _check_tree_locked(self):
       if hasattr(self, '_tree_locked') and self._tree_locked:
           raise RuntimeError("Cannot modify tree structure after it has been locked")
   ```

3. **Safe Operations on Locked Trees**:
   - Reading node content
   - Traversing relationships
   - Accessing metadata
   - Computing statistics
   - Visualization operations

4. **Error Handling**:
   - Clear error messages for modification attempts
   - Logging of unauthorized modification attempts
   - Proper exception propagation

5. **Performance Considerations**:
   - Lock checks add minimal overhead
   - Lock state stored in memory
   - No database queries for lock verification

##### Tree Data Format:
```json
{
    "tree": {
        "nodes": [
            {
                "id": "uuid",
                "text": "content",
                "level": 0,
                "metadata": {
                    "is_leaf": false,
                    "node_type": "root",
                    "tree_locked": true,
                    "lock_time": "timestamp"
                }
            }
        ]
    }
}
```

#### 5.4 Visualization System
- Interactive HTML reports with Plotly charts
- Real-time tree structure visualization
- Document distribution analysis
- Cluster relationship network
- System monitoring dashboard

##### Visualization Types
1. **Network Visualization**
   - Interactive force-directed graph
   - Node colors based on tree level
   - Node size reflects document count/importance
   - Edge thickness shows relationship strength

2. **Sunburst Diagram**
   - Hierarchical visualization showing tree structure
   - Center: Root node (highest level)
   - Concentric rings: Each level of hierarchy
     - Inner ring: Branches (level 2)
     - Middle ring: Twigs (level 1)
     - Outer ring: Document chunks (level 0)
   - Segment size represents number of descendants
   - Color coding matches tree levels:
     - Level 0 (leaves): Document chunks
     - Level 1: Twig nodes
     - Level 2: Branch nodes
     - Level 3: Root node

3. **Tree Health Dashboard**
   - Document distribution metrics
   - Node connectivity analysis
   - Level balance visualization
   - Quality metrics summary

##### Implementation Details
```python
# Color scheme for different tree levels
branch_colors = {
    0: "#2E86C1",  # Document chunks - blue
    1: "#27AE60",  # Twigs - green
    2: "#F39C12",  # Branches - orange
    3: "#C0392B"   # Root - red
}

# Node size configuration
node_size_range = {
    0: (15, 30),   # Document chunks
    1: (20, 40),   # Twigs
    2: (30, 50),   # Branches
    3: (40, 60)    # Root
}
```

### 6. System Monitoring

#### 6.1 Metrics Tracked
- Document processing metrics (total, processed, failed)
- Vector operations (cache performance, distribution)
- System resources (CPU, memory, disk)
- Network I/O statistics

#### 6.2 Monitoring Dashboard
Run the monitoring dashboard:
```bash
python -m src.scripts.monitor_system --base-dir /path/to/project --refresh-interval 30
```
Dashboard available at: `data/visualizations/dashboard.html`

### 7. Best Practices

#### 7.1 Environment Setup
- Never modify existing .env file
- Use command line for index names
- Validate environment before initialization

#### 7.2 Error Handling
- Check logs for initialization errors
- Verify Pinecone connection
- Monitor token usage

#### 7.3 Data Management
- Use consistent namespacing
- Verify data persistence
- Monitor storage usage

#### 7.4 Code Modification
Protected core components requiring special care:
- src/tree/tree_manager.py
- src/tree/raptor_tree.py
- src/storage/pinecone_manager.py
- Core managers in clustering/, embedding/, summarization/

Rules for modifying protected components:
1. Create feature branch
2. Full test coverage required
3. Code review mandatory
4. Backup working version
5. Document all changes

### 8. Troubleshooting Guide

#### 8.1 Initialization Failures
- Check environment variables
- Verify Pinecone index exists
- Confirm OpenAI API key validity

#### 8.2 Processing Issues
- Monitor chunking boundaries
- Check summarization output
- Verify embedding dimensions

#### 8.3 Storage Issues
- Confirm Pinecone connection
- Check namespace permissions
- Verify vector dimensions

### 9. Analysis Reports

The system generates comprehensive analysis reports in HTML format with:
1. Summary Section
   - Tree structure overview
   - Key metrics interpretation
   - Quality assessment
   - Improvement recommendations

2. Documents Section
   - Document collection analysis
   - Characteristics and distribution
   - Quality assessment
   - Interactive visualizations

3. Tree Visualization Section
   - Interactive tree structure
   - Cluster quality assessment
   - Depth and balance analysis
   - Metric interpretation

Generate a report:
```bash
python -m src.scripts.generate_report --index-name your-index-name
```
Reports are saved in: `reports/{index_name}/analysis_report_TIMESTAMP.html`

### 10. Important Notes

1. Never Create Duplicates Of:
   - Tree Management (keep in src/tree/)
   - Storage Operations (use pinecone_manager.py)
   - Core Managers (single instance per type)
   - Configuration (single source in config.py)

2. Data Flow:
   - Documents → Chunks → Embeddings → Tree → Pinecone
   - Each step has verification
   - Automatic error recovery

3. Document Processing:
   - Only documents in raw/ are processed
   - Each index maintains its own state
   - Never mix documents between indices
   - Failed documents are logged
   - Visualization failures don't affect processing

4. Configuration Chain:
```
Command Line (index_name)
↓
initialize_components
├── PineconeManager (singleton)
├── EmbedManager
├── OpenAI Client
└── TreeManager
    └── RaptorTree
```

Remember:
- Always use command line for index names
- Never modify existing .env
- Check logs for detailed error messages
- Verify Pinecone storage after processing
- Keep backups before major changes
- Test thoroughly in isolation
- Document all modifications

# Tree Structure

The system uses a dynamic tree structure based on the logarithm (base 20) of the number of chunks:

## Depth Levels

1. **Single Level** (1-2 chunks)
   - Root only
   
2. **Two Levels** (3-20 chunks)
   - Root
   - Leaves

3. **Three Levels** (21-400 chunks)
   - Root
   - Twigs (summary nodes)
   - Leaves

4. **Four Levels** (401-8000 chunks)
   - Root
   - Branches (high-level summaries)
   - Twigs (detailed summaries)
   - Leaves

5. **Five Levels** (8001+ chunks)
   - Root
   - Boughs (top-level summaries)
   - Branches (high-level summaries)
   - Twigs (detailed summaries)
   - Leaves

## Node Distribution

- Target ~20 children per node for optimal visualization
- Minimum 2 children per node for meaningful summaries
- Balanced distribution using geometric progression
- Automatic adjustment based on total document count

## Visualization

The tree structure can be visualized using:
1. Network graph (force-directed layout)
2. Sunburst diagram (hierarchical view)
3. Tree health dashboard (structure analytics)

## Best Practices

1. Keep document chunks relatively uniform in size
2. Aim for 300-500 tokens per chunk for optimal summarization
3. Monitor tree balance metrics in health dashboard
4. Use tree visualization to identify structural issues
