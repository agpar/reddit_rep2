from machine_learning.data_balance import *


# Methods to setup an adjacency matrix
def get_undirected_edges(node):
    node_edges = []
    for child in node.children:
        node_edges.append([node.index, child.index])
        node_edges.append([child.index, node.index])
    return node_edges


def get_directed_edges(node):
    node_edges = []
    for child in node.children:
        node_edges.append([node.index, child.index])
    return node_edges


# helper method to create an adjacency matrix given a tree of RedditNodes
def get_adj_matrix(
        tree,  # RedditTree to process
        is_undirected=True,
        has_parent_loop=False,  # If the root node has an edge to itself
        has_parent_edges=True,  # If the parent's children are connected to the parent
):
    node_edges = []
    for node in tree:
        index = node.index
        node_edges.append([index, index])  # a node is its own neighbour
        if (is_undirected):
            new_edges = get_undirected_edges(node)
        else:
            new_edges = get_directed_edges(node)
        node_edges.extend(new_edges)

    if (not has_parent_loop):
        # Use assumption that first edge is the parent loop
        node_edges = node_edges[1:]

    # Remove all edges directed to the parent node
    if (not has_parent_edges):
        node_edges = list(filter(lambda edge: edge[1] != 0, node_edges))

    edges = torch.LongTensor(node_edges)
    ones = torch.ones(edges.size(0))
    adj_matrix = torch.sparse.IntTensor(edges.t(), ones)
    return adj_matrix





