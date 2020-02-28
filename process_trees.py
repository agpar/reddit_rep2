import pickle
#
import torch
from torchtext.data.utils import get_tokenizer
#
from reddit_data import RedditTreeBuilder, JsonDataSource
from machine_learning.glove_embedding import UNKNOWN_WORD, glove, word2idx

tokenizer = get_tokenizer("basic_english")


# helper method to create an adjacency matrix given a tree of RedditNodes
def get_adj_matrix(tree):
    node_edges = []
    for node in tree:
        index = node.index
        node_edges.append([index, index]) # a node is its own neighbour
        for child in node.children:
            node_edges.append([index, child.index])

    edges = torch.LongTensor(node_edges)
    ones = torch.ones(edges.size(0))
    adj_matrix = torch.sparse.IntTensor(edges.t(), ones)
    return adj_matrix


def get_tree_text(tree):
    return [node.comment.body for node in tree]


def get_text_embedding(text):
    return torch.tensor([(word2idx[token] if token in glove else word2idx[UNKNOWN_WORD]) for token in tokenizer(text)])


def get_tree_text_embedding(texts):
    text_lst = [get_text_embedding(text) for text in texts]
    return text_lst


# Returns the output to predict for all Reddit trees
# In this case, it is binary if scores are above or equal to 1, the default Reddit score
def get_output(tree):
    return torch.FloatTensor([node.comment.score >= 1 for node in tree])


def get_tree_values(tree):
    adj_mat = get_adj_matrix(tree)
    texts = get_tree_text(tree)
    embedding = get_tree_text_embedding(texts)
    outputs = get_output(tree)

    return {
        "adj_matrix": adj_mat,
        "embedding": embedding,
        "outputs": outputs
    }


# Combine the results of multiple trees to a large graph
# trees is a list of dictionaries with preprocessed tree data
def combine_tree_values(trees):
    # Combine text embeddings and save offsets
    text_embeddings = [embedding for tree in trees for embedding in tree["embedding"]]

    offsets = [0] + [len(entry) for entry in text_embeddings]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    concat_text = torch.cat(text_embeddings)

    # Construct combined adjacency matrix
    num_nodes = [0] + [len(tree["outputs"]) for tree in trees]
    edges = torch.cat(
        [tree["adj_matrix"].coalesce().indices() + num_nodes[i] for i, tree in enumerate(trees)],
        dim=1
    )
    ones = torch.ones(edges.size(1))
    adj_matrix = torch.sparse.IntTensor(edges, ones)

    # Concatenate all outputs
    outputs = torch.cat([tree["outputs"] for tree in trees])

    model_data = {
        "adj_matrix": adj_matrix,
        "embedding": concat_text,
        "offsets": offsets,
        "outputs": outputs
    }
    return model_data


# if __name__ == "__main__":
# Load reddit data
# path_to_json_data = "reddit_data/RC_2006-12" # for example, use comments from 2006
# jds = JsonDataSource(path_to_json_data)
# rt = RedditTreeBuilder(jds)
# 
# all_roots = list(jds.get_roots())
# all_trees = [rt.get_tree_rooted_at(c) for c in all_roots]

with open("tree_test.pkl", 'rb') as input:
    tree_data = pickle.load(input)

# tree = all_trees[2]

# adj_mat = get_adj_matrix(tree)
# texts = get_tree_text(tree)
# embedding, offsets = get_tree_text_embedding(texts)
# outputs = get_output(tree)
data = combine_tree_values(tree_data)

# pickle.dump(model_data, open('machine_learning/sample_data.pkl', 'wb'))
