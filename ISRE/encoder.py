class ASTEncoder:
    """
    Tree neural network encoder (Tree GRU).

    Produces node embeddings and global context from AST.
    """

    def encode(self, ast):
        raise NotImplementedError