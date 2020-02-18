import pickle
import bcolz
import torch
import torch.nn as nn
import numpy as np
from reddit_data import RedditTrees, JsonDataSource
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

# 1. Get json formatted reddit data from https://files.pushshift.io/reddit/comments/
# 2. Use filter_subreddits.py to produce a single file containing only the subreddits you care about.
# 3. Look at example below on how to load a filtered json file and iterate over trees of comments.

path_to_json_data = "reddit_data/RC_2006-12" # for example, use comments from 2006
jds = JsonDataSource(path_to_json_data)
rt = RedditTrees(jds)

all_roots = list(jds.get_roots())
all_trees = [rt.get_tree_rooted_at(c) for c in all_roots]

glove_path = "machine_learning/glove"

word_vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

glove = {w: word_vectors[word2idx[w]] for w in words}

target_vocab = words

matrix_len = len(target_vocab)
weights_matrix = np.zeros((matrix_len, 50))
words_found = 0

for i, word in enumerate(target_vocab):
    try:
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))

class TextSentiment(nn.Module):
    def __init__(self, embedding_weight, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(embedding_weight)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

# torch.tensor([word2idx[token] for token in ngrams_iterator(tokenizer(text), 1)])
