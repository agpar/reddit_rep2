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


def combine_bow(trees):
    # Concatenate bow values
    bag_of_words = torch.cat([tree["bag_of_words"] for tree in trees])
    return bag_of_words


# Combine the results of multiple trees to a large graph
# trees is a list of dictionaries with preprocessed tree data
def combine_bow_tree_values(trees):
    bag_of_words = combine_bow(trees)
    adj_matrix = combine_adj_matrix(trees)
    outputs = combine_outputs(trees)

    parent_indices = [0] + [len(entry["outputs"]) for entry in trees]
    parent_indices = torch.tensor(parent_indices[:-1]).cumsum(dim=0)

    model_data = {
        "adj_matrix": adj_matrix,
        "bag_of_words": bag_of_words,
        "outputs": outputs,
        "parent_indices": parent_indices
    }
    return model_data