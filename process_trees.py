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


# Using an EmbeddingBag, the api requires text to be concatenated and offsets for each text to be specified
def get_tree_text_embedding(texts):
    text_lst = [get_text_embedding(text) for text in texts]
    offsets = [0] + [len(entry) for entry in text_lst]

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    concat_text = torch.cat(text_lst)
    return concat_text, offsets


# Returns the output to predict for all Reddit trees
# In this case, it is binary if scores are above or equal to 1, the default Reddit score
def get_output(tree):
    return torch.FloatTensor([node.comment.score >= 1 for node in tree])


if __name__ == "__main__":
    # Load reddit data
    path_to_json_data = "reddit_data/RC_2006-12" # for example, use comments from 2006
    jds = JsonDataSource(path_to_json_data)
    rt = RedditTreeBuilder(jds)

    all_roots = list(jds.get_roots())
    all_trees = [rt.get_tree_rooted_at(c) for c in all_roots]

    tree = all_trees[2]

    adj_mat = get_adj_matrix(tree)
    texts = get_tree_text(tree)
    embedding, offsets = get_tree_text_embedding(texts)
    outputs = get_output(tree)

    model_data = {
        "adj_matrix": adj_mat,
        "embedding": embedding,
        "offsets": offsets,
        "outputs": outputs
    }
    pickle.dump(model_data, open('machine_learning/sample_data.pkl', 'wb'))
