import random

import torch
from profanity_check import predict
from textblob import TextBlob

from machine_learning.experiment_types import *
from machine_learning.language_collections import hate_dict


def get_scores(tree):
    return [node.comment.score for node in tree]


def get_tree_text(tree):
    return [node.comment.body for node in tree]


def get_score_polarity(tree):
    return torch.LongTensor([(0 if node.comment.score >= 1 else 1) for node in tree])


def get_has_profanity(tree):
    texts = get_tree_text(tree)
    return torch.LongTensor(predict(texts))


def get_sentiment(tree):
    texts = get_tree_text(tree)
    sentiments = [TextBlob(text).sentiment.polarity for text in texts]
    return torch.LongTensor([(0 if score >= 0 else 1) for score in sentiments])


def has_hate_speech(text):
    blob = TextBlob(text)
    just_text = " ".join(blob.words).lower()
    return any(hate in just_text for hate in hate_dict.keys())


def get_hate_speech(tree):
    texts = get_tree_text(tree)
    return torch.LongTensor([(1 if has_hate_speech(text) else 0) for text in texts])


def get_subreddit(tree):
    return torch.LongTensor([subreddit_map[node.comment.subreddit] for node in tree])


def get_random_output(tree):
    return torch.LongTensor([random.randint(0, 1) for node in tree])


def get_output(tree, output_type: OutputType):
    if (output_type == OutputType.SCORE_POLARITY):
        return get_score_polarity(tree)
    elif (output_type == OutputType.PROFANITY):
        return get_has_profanity(tree)
    elif (output_type == OutputType.SENTIMENT):
        return get_sentiment(tree)
    elif (output_type == OutputType.HATE_SPEECH):
        return get_hate_speech(tree)
    elif (output_type == OutputType.SUBREDDIT):
        return get_subreddit(tree)
    elif (output_type == OutputType.RANDOM):
        return get_random_output(tree)
    else:
        raise Exception("Invalid output type")
