import os
import pickle

import bcolz
import torch
from torchtext.data.utils import get_tokenizer
from nltk.corpus import stopwords
import numpy as np

from machine_learning.feature_extraction import get_tree_text

UNKNOWN_WORD = "[unk]"

_tokenizer = get_tokenizer("spacy")
_eng_stop_words = set(_tokenizer(" ".join(stopwords.words("english")).replace("'","")))


class GloveEmbedder:
    def __init__(self, glove_path, dim):
        self.glove_path = glove_path
        self.dim = dim
        self.word_vectors = None
        self.words = None
        self.word2idx = None
        self.glove = None

    def setup(self):
        words = []
        word2idx = {}
        vectors = bcolz.carray(np.zeros(1), rootdir=f'{self.glove_path}/6B.50.dat', mode='w')

        with open(f'{self.glove_path}/glove.6B.50d.txt', 'rb') as file:
            for idx, line in enumerate(file):
                line_arr = line.decode().split()
                word = line_arr[0]
                words.append(word)
                word2idx[word] = idx
                word_vec = np.array(line_arr[1:]).astype(np.float)
                vectors.append(word_vec)

        # Add an extra word for unknown vocab. It is the mean of all vectors
        # https://stackoverflow.com/questions/49239941/what-is-unk-in-the-pretrained-glove-vector-files-e-g-glove-6b-50d-txt
        avg_vec = np.mean(vectors[1:].reshape((400001, 50)), axis=0)
        unknown_word = "[unk]" # Note: "unk" and "<unk>" are actually words in GloVe

        word2idx[unknown_word] = len(words)
        words.append(unknown_word)
        vectors.append(avg_vec)

        # Save the results
        vectors = bcolz.carray(vectors[1:].reshape((400002, 50)), rootdir=f'{self.glove_path}/6B.{self.dim}.dat', mode='w')
        vectors.flush()
        pickle.dump(words, open(f'{self.glove_path}/6B.{self.dim}_words.pkl', 'wb'))
        pickle.dump(word2idx, open(f'{self.glove_path}/6B.{self.dim}_idx.pkl', 'wb'))

    def load(self):
        test_path = f'{self.glove_path}/6B.{self.dim}.dat'
        if not os.path.exists(test_path):
            raise Exception("fGLOVE .dat file does not exit. Either check the glove_path or run setup(): {test_path}")
        self.word_vectors = bcolz.open(f'{self.glove_path}/6B.{self.dim}.dat')[:]
        self.words = pickle.load(open(f'{self.glove_path}/6B.{self.dim}_words.pkl', 'rb'))
        self.word2idx = pickle.load(open(f'{self.glove_path}/6B.{self.dim}_idx.pkl', 'rb'))
        self.glove = {w: self.word_vectors[self.word2idx[w]] for w in self.words}

    def get_text_embedding(self, text):
        return torch.tensor([(self.word2idx[token] if token in self.glove
                              else self.word2idx[UNKNOWN_WORD]) for token in _tokenizer(text)])

    def get_tree_text_embedding(self, texts):
        text_lst = [self.get_text_embedding(text) for text in texts]
        return text_lst

    def get_glove_embedding(self, tree):
        texts = get_tree_text(tree)
        embedding = self.get_tree_text_embedding(texts)
        return embedding
