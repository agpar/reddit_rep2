import random

from machine_learning.experiment_types import *
from machine_learning.data_balance import balance_trees
from machine_learning.adjacency_matrix import get_adj_matrix
from machine_learning.bow_embedding import BowEmbedder
from machine_learning.glove_embedding import GloveEmbedder
from machine_learning.feature_extraction import get_output
from machine_learning.batch_combiner import combine_bow_tree_values, combine_glove_tree_values


# Methods for outputs
def get_model_data(
        trees,
        batch_size,
        output_type: OutputType,
        vectorizer_type: VectorizerType,
        is_undirected=True,
        has_parent_loop=False,  # If the root node has an edge to itself
        has_parent_edges=True,  # If the parent's children are connected to the parent
):
    # Balance the trees
    balanced_trees = balance_trees(trees, output_type)
    # Create adjacency matrices
    adj_matrices = [get_adj_matrix(tree, is_undirected, has_parent_loop, has_parent_edges) for tree in balanced_trees]
    # Create inputs
    if vectorizer_type in (VectorizerType.COUNT, VectorizerType.TFIDF):
        embeddings, _ = BowEmbedder().get_bow_embedding(balanced_trees, vectorizer_type)
    else:
        embedder = GloveEmbedder("/content/drive/My Drive/glove", 50)
        embeddings = [embedder.get_glove_embedding(tree) for tree in balanced_trees]
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
    # Split into train and balanced valid set
    # Batch trees together
    if vectorizer_type in (VectorizerType.COUNT, VectorizerType.TFIDF):
        model_data = [
            combine_bow_tree_values(tree_data[i:(i + batch_size)])
            for i in range(0, len(tree_data), batch_size)
        ]
    else:
        model_data = [
            combine_glove_tree_values(tree_data[i:(i + batch_size)])
            for i in range(0, len(tree_data), batch_size)
        ]

    return model_data
