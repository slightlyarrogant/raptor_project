"""Text chunk class for managing text segments."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class TextChunk:
    """A chunk of text with associated metadata and analysis."""
    
    text: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: str = ""
    main_topic: Optional[str] = None
    key_concepts: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    summary_metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate and initialize the chunk."""
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        if not isinstance(self.start_index, int):
            raise TypeError("start_index must be an integer")
        if not isinstance(self.end_index, int):
            raise TypeError("end_index must be an integer")
        if self.chunk_id is not None and not isinstance(self.chunk_id, str):
            raise TypeError("chunk_id must be a string or None")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'text': self.text,
            'metadata': self.metadata,
            'context': self.context,
            'main_topic': self.main_topic,
            'key_concepts': self.key_concepts,
            'dependencies': self.dependencies,
            'relationships': self.relationships,
            'summary_metadata': self.summary_metadata,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "chunk_id": self.chunk_id
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextChunk':
        """Create from dictionary representation."""
        return cls(
            text=data['text'],
            metadata=data.get('metadata', {}),
            context=data.get('context', ''),
            main_topic=data.get('main_topic'),
            key_concepts=data.get('key_concepts', []),
            dependencies=data.get('dependencies', []),
            relationships=data.get('relationships', []),
            summary_metadata=data.get('summary_metadata', {}),
            start_index=data["start_index"],
            end_index=data["end_index"],
            chunk_id=data.get("chunk_id")
        )
