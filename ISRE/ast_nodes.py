from dataclasses import dataclass
from typing import List


@dataclass
class ASTNode:
    node_type: str
    children: List["ASTNode"]
    depth: int
    subtree_size: int


@dataclass
class CandidateAction:
    node_id: int
    action_type: str
