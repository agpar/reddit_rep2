from enum import Enum


class OutputType(Enum):
    SCORE_POLARITY = 1
    PROFANITY = 2
    SENTIMENT = 3
    HATE_SPEECH = 4
    SUBREDDIT = 5
    RANDOM = 6


class VectorizerType(Enum):
    COUNT = 1
    TFIDF = 2


subreddit_map = {
    "movies": 0,
    "worldnews": 1,
    "AskReddit": 2
}
