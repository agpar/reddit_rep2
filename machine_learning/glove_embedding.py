import os
import bcolz
import pickle

UNKNOWN_WORD = "[unk]"

# Setup glove embedding
script_dir = os.path.dirname(__file__)
glove_path = os.path.join(script_dir, "glove")

word_vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

glove = {w: word_vectors[word2idx[w]] for w in words}
