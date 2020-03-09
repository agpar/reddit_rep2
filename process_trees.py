import pickle
#
import torch
from torchtext.data.utils import get_tokenizer
from sklearn.feature_extraction.text import CountVectorizer
#
from reddit_data import RedditTreeBuilder, JsonDataSource
from machine_learning.glove_embedding import UNKNOWN_WORD, glove, word2idx

tokenizer = get_tokenizer("basic_english")


# helper method to create an adjacency matrix given a tree of RedditNodes
def get_adj_matrix(tree):
    node_edges = []
    for node in tree:
        index = node.index
        # node_edges.append([index, index]) # a node is its own neighbour
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
    return torch.FloatTensor([([1, 0] if node.comment.score >= 1 else [0, 1]) for node in tree])


def get_bag_of_words(trees):
    texts = [get_tree_text(tree) for tree in trees]
    flat_texts = [text for tree_text in texts for text in tree_text]

    vectorizer = CountVectorizer(stop_words="english")
    word_freq = vectorizer.fit_transform(flat_texts)

    num_nodes = [0] + [len(text) for text in texts]
    num_nodes = torch.tensor(num_nodes).cumsum(dim=0)
    tensor_word_freq = [torch.tensor(word_freq[num_nodes[i]:num_nodes[i + 1]].toarray()) for i in range(len(texts))]

    return tensor_word_freq, vectorizer


def get_glove_tree_values(tree):
    adj_mat = get_adj_matrix(tree)
    texts = get_tree_text(tree)
    embedding = get_tree_text_embedding(texts)
    outputs = get_output(tree)

    return {
        "adj_matrix": adj_mat,
        "embedding": embedding,
        "outputs": outputs
    }

def get_bow_tree_values(trees):
    bow_values, vectorizer = get_bag_of_words(trees)
    tree_values = []

    for i, tree in enumerate(trees):
        adj_matrix = get_adj_matrix(tree)

        outputs = get_output(tree)
        parent_indices = adj_matrix.coalesce().indices()[0].unique()
        outputs = outputs[parent_indices]

        tree_values.append({
            "adj_matrix": adj_matrix,
            "bag_of_words": bow_values[i],
            "outputs": outputs
        })
    return tree_values, vectorizer


def combine_outputs(trees):
    # Concatenate all outputs
    outputs = torch.cat([tree["outputs"] for tree in trees])
    return outputs

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

def combine_text_embedding(trees):
    # Combine text embeddings and save offsets
    text_embeddings = [embedding for tree in trees for embedding in tree["embedding"]]

    offsets = [0] + [len(entry) for entry in text_embeddings]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    concat_text = torch.cat(text_embeddings)

    return concat_text, offsets


# Combine the results of multiple trees to a large graph
# trees is a list of dictionaries with preprocessed tree data
def combine_glove_tree_values(trees):
    concat_text, offsets = combine_text_embedding(trees)
    adj_matrix = combine_adj_matrix(trees)

    outputs = combine_outputs(trees)
    parent_indices = adj_matrix.coalesce().indices()[0].unique()
    outputs = outputs[parent_indices]

    model_data = {
        "adj_matrix": adj_matrix,
        "embedding": concat_text,
        "offsets": offsets,
        "outputs": outputs
    }
    return model_data




# if __name__ == "__main__":
# Load reddit data
path_to_json_data = "reddit_data/RC_2006-12" # for example, use comments from 2006
jds = JsonDataSource(path_to_json_data)
rt = RedditTreeBuilder(jds)

all_roots = list(jds.get_roots())
all_trees = [rt.get_tree_rooted_at(c) for c in all_roots]

# only include trees that have at least one child
training_trees = list(filter(lambda tree: len(tree.children) > 0, all_trees))
# tree_data = [get_tree_values(tree) for tree in training_trees]
# 
# model_data = [combine_tree_values([tree]) for tree in tree_data]
# pickle.dump(model_data, open('machine_learning/sample_data.pkl', 'wb'))

with open("reddit_tree.pkl", 'rb') as input:
    tree = pickle.load(input)

# texts = get_tree_text(tree)
# vectorizer = CountVectorizer(stop_words="english")
# inputs = vectorizer.fit_transform(texts)
trees = training_trees[:5]
model_data, vectorizer = get_bow_tree_values(trees)

# texts = [get_tree_text(tree) for tree in trees]
# flat_texts = [text for tree_text in texts for text in tree_text]
# 
# vectorizer = CountVectorizer(stop_words="english")
# word_freq = vectorizer.fit_transform(flat_texts)
# 
# num_nodes = [0] + [len(text) for text in texts]
# num_nodes = torch.tensor(num_nodes).cumsum(dim=0)
# tensor_word_freq = torch.tensor(word_freq.toarray())
