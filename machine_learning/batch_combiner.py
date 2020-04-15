import torch


def combine_adj_matrix(trees):
    # Construct combined adjacency matrix
    num_nodes = [0] + [len(tree["outputs"]) for tree in trees]
    num_nodes = torch.tensor(num_nodes).cumsum(dim=0)
    edges = torch.cat(
        [tree["adj_matrix"].coalesce().indices() + num_nodes[i] for i, tree in enumerate(trees)],
        dim=1
    )
    ones = torch.ones(edges.size(1))
    adj_matrix = torch.sparse.IntTensor(edges, ones)
    return adj_matrix


def combine_outputs(trees):
    # Concatenate all outputs
    outputs = torch.cat([tree["outputs"] for tree in trees])
    return outputs


def combine_bow_tree_values(trees):
    bag_of_words = torch.cat([tree["embedding"] for tree in trees])
    adj_matrix = combine_adj_matrix(trees)
    outputs = combine_outputs(trees)

    parent_indices = [0] + [len(entry["outputs"]) for entry in trees]
    parent_indices = torch.tensor(parent_indices[:-1]).cumsum(dim=0)

    model_data = {
        "adj_matrix": adj_matrix,
        "embedding": bag_of_words,
        "offsets": None,
        "outputs": outputs,
        "parent_indices": parent_indices
    }
    return model_data


def combine_text_embedding(trees):
    # Combine text embeddings and save offsets
    text_embeddings = [embedding for tree in trees for embedding in tree["embedding"]]

    offsets = [0] + [len(entry) for entry in text_embeddings]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    concat_text = torch.cat(text_embeddings)

    return concat_text, offsets


def combine_glove_tree_values(trees):
    concat_text, offsets = combine_text_embedding(trees)
    adj_matrix = combine_adj_matrix(trees)
    outputs = combine_outputs(trees)

    parent_indices = [0] + [len(entry["outputs"]) for entry in trees]
    parent_indices = torch.tensor(parent_indices[:-1]).cumsum(dim=0)

    model_data = {
        "adj_matrix": adj_matrix,
        "embedding": concat_text,
        "offsets": offsets,
        "outputs": outputs,
        "parent_indices": parent_indices
    }
    return model_data
