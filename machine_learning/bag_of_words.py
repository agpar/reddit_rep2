import nltk
import torch
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torchtext.data.utils import get_tokenizer

from machine_learning.experiment_types import *
from machine_learning.feature_extraction import get_tree_text
from text_preprocessing import stemming_tokenizer

# Methods to extract words
nltk.download("stopwords")

_eng_stop_words = set(stemming_tokenizer(" ".join(stopwords.words("english")).replace("'", "")))
tokenizer = get_tokenizer("spacy")


def get_bag_of_words(trees, vectorizer_type: VectorizerType):
    texts = [get_tree_text(tree) for tree in trees]
    flat_texts = [text for tree_text in texts for text in tree_text]

    if (vectorizer_type == VectorizerType.COUNT):
        Vectorizer = CountVectorizer
    elif (vectorizer_type == VectorizerType.TFIDF):
        Vectorizer = TfidfVectorizer
    else:
        raise Exception("Invalid vectorizer type")

    vectorizer = Vectorizer(stop_words=_eng_stop_words, tokenizer=stemming_tokenizer, min_df=0.001)
    word_freq = vectorizer.fit_transform(flat_texts)

    num_nodes = [0] + [len(text) for text in texts]
    num_nodes = torch.tensor(num_nodes).cumsum(dim=0)
    tensor_word_freq = [torch.FloatTensor(word_freq[num_nodes[i]:num_nodes[i + 1]].toarray()) for i in
                        range(len(texts))]

    return tensor_word_freq, vectorizer
