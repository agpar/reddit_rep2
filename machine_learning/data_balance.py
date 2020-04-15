# Methods for balancing data
from random import sample, randint

import torch
from profanity_check import predict
from textblob import TextBlob

from machine_learning.experiment_types import OutputType, subreddit_map
from machine_learning.language_collections import hate_dict


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
    return torch.LongTensor([randint(0, 1) for node in tree])


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


# Balance the trees by score
def balance_by_score_polarity(trees):
    positive_trees = list(filter(lambda tree: tree.comment.score >= 1, trees))
    negative_trees = list(filter(lambda tree: tree.comment.score < 1, trees))

    undersample_positive = sample(positive_trees, len(negative_trees))
    training_trees = undersample_positive + negative_trees
    return training_trees


# Balance the trees by profanity
def balance_by_profanity(trees):
    profane_text = predict([tree.comment.body for tree in trees])

    positive_trees = [tree for i, tree in enumerate(trees) if profane_text[i] >= 1]
    negative_trees = [tree for i, tree in enumerate(trees) if profane_text[i] < 1]

    undersample_negative = sample(negative_trees, len(positive_trees))
    training_trees = undersample_negative + positive_trees
    return training_trees


# Balance trees by sentiment
def balance_by_sentiment(trees):
    positive_trees = []
    negative_trees = []
    for tree in trees:
        if (TextBlob(tree.comment.body).sentiment.polarity >= 0):
            positive_trees.append(tree)
        else:
            negative_trees.append(tree)

    undersample_positive = sample(positive_trees, len(negative_trees))
    training_trees = undersample_positive + negative_trees
    return training_trees


# Balance trees by hate speech
def balance_by_hate_speech(trees):
    positive_trees = []
    negative_trees = []
    for tree in trees:
        if (has_hate_speech(tree.comment.body)):
            positive_trees.append(tree)
        else:
            negative_trees.append(tree)

    undersample_negative = sample(negative_trees, len(positive_trees))
    training_trees = undersample_negative + positive_trees
    return training_trees


# Balance by subreddit
def balance_by_subreddit(trees):
    movie_trees = list(filter(lambda tree: tree.comment.subreddit == "movies", trees))
    world_news_trees = list(filter(lambda tree: tree.comment.subreddit == "worldnews", trees))
    ask_reddit_trees = list(filter(lambda tree: tree.comment.subreddit == "AskReddit", trees))

    class_size = min(len(movie_trees), len(world_news_trees), len(ask_reddit_trees))
    training_trees = sample(movie_trees, class_size) + \
                     sample(world_news_trees, class_size) + \
                     sample(ask_reddit_trees, class_size)
    return training_trees


# Balance to be 1/10 of original size
def balance_by_random(trees):
    sample_size = len(trees) // 10
    training_trees = sample(trees, sample_size)
    return training_trees


# General balance function
def balance_trees(trees, output_type: OutputType):
    if (output_type == OutputType.SCORE_POLARITY):
        return balance_by_score_polarity(trees)
    elif (output_type == OutputType.PROFANITY):
        return balance_by_profanity(trees)
    elif (output_type == OutputType.SENTIMENT):
        return balance_by_sentiment(trees)
    elif (output_type == OutputType.HATE_SPEECH):
        return balance_by_hate_speech(trees)
    elif (output_type == OutputType.SUBREDDIT):
        return balance_by_subreddit(trees)
    elif (output_type == OutputType.RANDOM):
        return balance_by_random(trees)
    else:
        raise Exception("Invalid output type")
