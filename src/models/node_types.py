from dataclasses import dataclass, field
from typing import Dict, List, Union

@dataclass
class LeafNode:
    """Document chunk node containing raw text."""
    id: str
    text: str
    level: int = 0
    metadata: Dict = field(default_factory=dict)
    children: List = field(default_factory=list)

@dataclass
class SummaryNode:
    """Internal node containing summarized content."""
    id: str
    text: str
    level: int
    metadata: Dict = field(default_factory=dict)
    children: List['Union[LeafNode, SummaryNode]'] = field(default_factory=list) 