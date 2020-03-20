import string
#
import torch
from torchtext.data.utils import get_tokenizer
from nltk.stem.porter import PorterStemmer

tokenizer = get_tokenizer("basic_english")
puncTable = str.maketrans('', '', string.punctuation)
stemmer = PorterStemmer()

def stemming_tokenizer(text):
    words = [stemmer.stem(word.translate(puncTable)) for word in tokenizer(text)]
    words = list(filter(lambda word: word.isalpha(), words))
    return words

def get_parent_indices(adj):
    if (adj.type() == "torch.sparse.FloatTensor"):
        indices, counts = adj.coalesce().indices()[0].unique(return_counts=True)
    else:
        indices, counts = torch.nonzero(adj).t()[0].unique(return_counts=True)
    return indices[counts > 2]
