from collections import deque
import pickle
#
import torch
import bcolz
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
#
from reddit_data import RedditTrees, JsonDataSource

UNKNOWN_WORD = "[unk]"

# Setup glove embedding
glove_path = "machine_learning/glove"

word_vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

glove = {w: word_vectors[word2idx[w]] for w in words}
tokenizer = get_tokenizer("basic_english")

# Load reddit data
path_to_json_data = "reddit_data/RC_2006-12" # for example, use comments from 2006
jds = JsonDataSource(path_to_json_data)
rt = RedditTrees(jds)

all_roots = list(jds.get_roots())
all_trees = [rt.get_tree_rooted_at(c) for c in all_roots]

with open("reddit_tree.pkl", 'rb') as input:
    tree = pickle.load(input)

# Depth first search as a generator
def dfs(tree):
    queue = deque()
    queue.append(tree)
    while(queue):
        node = queue.popleft()
        yield node
        for child in node.children:
            queue.append(child)

# helper method to assign each node an index, and extract comment text
# def create_get_node_id(id_to_idx):
#     count = 0
#     def get_node_id(node):
#         nonlocal count
#         id_to_idx[node.comment.id] = count
#         count += 1
#     return process_node_id

def create_adj_matrix(edges, id_to_index):
    def add_edge(node):
        index = id_to_index[node.comment.id]
        for child in node.children:
            child_index = id_to_index[child.comment.id]
            edges.append([index, child_index])
    return add_edge

def get_tree_indices(tree):
    return {node.comment.id: index for index, node in enumerate(dfs(tree))}

# helper method to create an ajacency matrix given a mapping from each node to an index
# The id_to_index mapping is passed to simplify the logic
def get_adj_matrix(tree, id_to_index):
    node_edges = []
    for node in dfs(tree):
        index = id_to_index[node.comment.id]
        for child in node.children:
            child_index = id_to_index[child.comment.id]
            node_edges.append([index, child_index])

    edges = torch.LongTensor(node_edges)
    ones = torch.ones(edges.size(0))
    adj_matrix = torch.sparse.IntTensor(edges.t(), ones)
    return edges

def get_tree_text(tree):
    return [node.comment.body for node in dfs(tree)]

def get_text_embedding(text):
    return torch.tensor([(word2idx[token] if token in glove else word2idx[UNKNOWN_WORD]) for token in ngrams_iterator(tokenizer(text), 1)])

# Using an EmbeddingBag, the api requires text to be concatenated and offsets for each text to be specified
def get_tree_text_embedding(texts):
    text_lst = [get_text_embedding(text) for text in texts]
    offsets = [0] + [len(entry) for entry in text_lst]

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    concat_text = torch.cat(text_lst)
    return concat_text, offsets

# Returns a tuple of (adj_matrix, tensor(text_embedding)) for a given RedditTree
# def get_tree_representation(tree):
#     id_to_index = dict()
#     node_edges = []
# 
#     # Iterate through tree to extract text and graph structure
#     process_nodes = create_process_node_id(id_to_index, text_nodes)
#     dfs(tree, process_nodes)
#     process_edges = create_adj_matrix(node_edges, id_to_index)
#     dfs(tree, process_edges)
# 
#     # Generate adjacency matrix as sparse tensor
#     edges = torch.LongTensor(node_edges)
#     ones = torch.ones(edges.size(0))
#     adj_matrix = torch.sparse.IntTensor(edges.t(), ones)
# 
#     # Convert text into embedding
#     text_tensors = [torch.tensor([(word2idx[token] if token in glove else word2idx[UNKNOWN_WORD])
#         for token in ngrams_iterator(tokenizer(text), 1)]) for text in text_nodes]
# 
#     return (adj_matrix, text_tensors)

# Returns the output to predict for all Reddit trees
# In this case, it is binary if scores are above or equal to 1, the default Reddit score
def get_output(trees):
    return torch.tensor([tree.comment.score >= 1 for tree in trees])

tree_indices = get_tree_indices(tree)
adj_mat = get_adj_matrix(tree, tree_indices)
texts = get_tree_text(tree)
embedding, offsets = get_tree_text_embedding(texts)

