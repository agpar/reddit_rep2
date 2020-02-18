from reddit_data import RedditTrees, JsonDataSource
from collections import deque
import pickle
import torch
import bcolz
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

# 1. Get json formatted reddit data from https://files.pushshift.io/reddit/comments/
# 2. Use filter_subreddits.py to produce a single file containing only the subreddits you care about.
# 3. Look at example below on how to load a filtered json file and iterate over trees of comments.

path_to_json_data = "reddit_data/RC_2006-12" # for example, use comments from 2006
# jds = JsonDataSource(path_to_json_data)
# rt = RedditTrees(jds)
# 
# all_roots = list(jds.get_roots())
# all_trees = [rt.get_tree_rooted_at(c) for c in all_roots]

#tree = all_trees[2]
with open("reddit_tree.pkl", 'rb') as input:
    tree = pickle.load(input)

def dfs(tree, func):
    queue = deque()
    queue.append(tree)
    while(queue):
        node = queue.popleft()
        func(node)
        for child in node.children:
            queue.append(child)

id_to_index = dict()
text_nodes = []
def create_process_node_id(id_to_idx, text_tensors):
    count = 0
    def process_node_id(node):
        nonlocal count
        id_to_idx[node.comment.id] = count
        count += 1
        text = node.comment.body
        text_tensors.append(text)
    return process_node_id

node_edges = []
def create_adj_matrix(edges, id_to_index):
    def add_edge(node):
        index = id_to_index[node.comment.id]
        for child in node.children:
            child_index = id_to_index[child.comment.id]
            edges.append([index, child_index])
    return add_edge

process_nodes = create_process_node_id(id_to_index, text_nodes)
dfs(tree, process_nodes)
process_edges = create_adj_matrix(node_edges, id_to_index)
dfs(tree, process_edges)

edges = torch.LongTensor(node_edges)
ones = torch.ones(edges.size(0))
adj_matrix = torch.sparse.IntTensor(edges.t(), ones)

glove_path = "machine_learning/glove"

word_vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

glove = {w: word_vectors[word2idx[w]] for w in words}
tokenizer = get_tokenizer("basic_english")
text_tensors = [torch.tensor([word2idx[token] for token in ngrams_iterator(tokenizer(text), 1)]) for text in text_nodes]


# torch.tensor([word2idx[token] for token in ngrams_iterator(tokenizer(text), 1)])
