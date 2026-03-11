from typing import List
from ast import ASTNode, CandidateAction


class SymbolicEngine:
    """
    Deterministic module that generates valid algebraic transformations
    for a given AST.
    """

    def generate_candidates(self, ast: ASTNode) -> List[CandidateAction]:
        raise NotImplementedError("Candidate generation not implemented yet.")

    def apply_transform(self, ast: ASTNode, action: CandidateAction) -> ASTNode:
        raise NotImplementedError("Transform application not implemented yet.")
    