import random
import torch

from machine_learning.experiment_types import *
from machine_learning.data_balance import balance_trees
from machine_learning.adjacency_matrix import get_adj_matrix
from machine_learning.feature_extraction import get_output, get_scores
from machine_learning.text_embedders import TextEmbedder

def append_scores(trees, embedding, mask_parent=True):
    scores = [torch.tensor(get_scores(tree)) for tree in trees]
    combined_embedding = []
    for i, score in enumerate(scores):
        if (mask_parent):
            score[0] = 0
        reshaped_score = score.view(len(score), 1).float()
        combined_embedding.append(torch.cat((embedding[i], reshaped_score), dim=1))
    return combined_embedding


# Methods for outputs
def get_model_data(
        trees,
        embedder: TextEmbedder,
        batch_size,
        output_type: OutputType,
        has_scores = True, # Whether the embedding has comment scores appended
        is_undirected=True,
        has_parent_loop=False,  # If the root node has an edge to itself
        has_parent_edges=True,  # If the parent's children are connected to the parent
        has_child_edges = True, # If the children are connected to the parent
):
    # Balance the trees
    balanced_trees = balance_trees(trees, output_type)
    # Create adjacency matrices
    adj_matrices = [get_adj_matrix(
        tree,
        is_undirected,
        has_parent_loop,
        has_parent_edges,
        has_child_edges,
    ) for tree in balanced_trees]
    # Create inputs
    embeddings = embedder.get_embeddings(balanced_trees)
    # Optionally append scores
    if (has_scores):
        embeddings = append_scores(
            balanced_trees,
            embeddings,
            mask_parent=output_type == OutputType.SCORE_POLARITY
        )
    # Create outputs
    outputs = [get_output(tree, output_type) for tree in balanced_trees]
    # Zip all data together
    tree_data = [{
        "adj_matrix": adj_mat,
        "embedding": emb,
        "outputs": output
    } for adj_mat, emb, output in zip(adj_matrices, embeddings, outputs)]
    # Scramble the data
    random.shuffle(tree_data)
    # Batch trees together
    return [
        embedder.batch_trees(tree_data[i:(i + batch_size)])
        for i in range(0, len(tree_data), batch_size)
    ]
