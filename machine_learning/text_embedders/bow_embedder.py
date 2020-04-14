import nltk
import torch
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torchtext.data.utils import get_tokenizer

from machine_learning.experiment_types import *
from machine_learning.feature_extraction import get_tree_text
from machine_learning.batch_combiner import combine_bow_tree_values
from machine_learning.text_embedders.text_embedder import TextEmbedder

nltk.download("stopwords")
_tokenizer = get_tokenizer("spacy")
_eng_stop_words = set(_tokenizer(" ".join(stopwords.words("english")).replace("'", "")))


class BowEmbedder(TextEmbedder):
    def __init__(self, vectorizer_type):
        self.vectorizer_type = vectorizer_type

    def get_embeddings(self, trees):
        return self.get_bow_embedding(trees)[0]

    def batch_trees(self, tree_data):
        return combine_bow_tree_values(tree_data)

    def get_bow_embedding(self, trees):
        texts = [get_tree_text(tree) for tree in trees]
        flat_texts = [text for tree_text in texts for text in tree_text]

        if (self.vectorizer_type == VectorizerType.COUNT):
            Vectorizer = CountVectorizer
        elif (self.vectorizer_type == VectorizerType.TFIDF):
            Vectorizer = TfidfVectorizer
        else:
            raise Exception("Invalid vectorizer type")

        vectorizer = Vectorizer(stop_words=_eng_stop_words, tokenizer=_tokenizer, min_df=0.001)
        word_freq = vectorizer.fit_transform(flat_texts)

        num_nodes = [0] + [len(text) for text in texts]
        num_nodes = torch.tensor(num_nodes).cumsum(dim=0)
        tensor_word_freq = [torch.FloatTensor(word_freq[num_nodes[i]:num_nodes[i + 1]].toarray()) for i in
                            range(len(texts))]

        return tensor_word_freq, vectorizer
