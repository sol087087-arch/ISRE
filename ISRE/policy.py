class PolicyNetwork:
    """
    Scores candidate symbolic transformations.

    Input:
        node embeddings + global context
        candidate actions

    Output:
        score(node, action)
    """

    def score(self, node_embedding, action_embedding):
        raise NotImplementedError