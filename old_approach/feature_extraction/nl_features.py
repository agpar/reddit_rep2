"""
Natural language features defind on a single Comment.
"""

import logging

from polyglot.detect import Detector
from textblob import TextBlob
from profanity_check import predict_prob as profanity
from hatesonar import Sonar

from old_approach.feature_extraction.comment_features import CommentFeatures
from old_approach.feature_extraction.nl_sets import *
from reddit_data_interface.data_models import RedditComment
from machine_learning.language_collections import hate_dict

logging.basicConfig(filename='./output.log')
hatesonar = Sonar()


def compute_nl_features(c: RedditComment):
    c.feats = CommentFeatures()
    stats = c.feats

    stats['lang'] = comment_languge(c)
    stats['word_count'] = word_count(c)
    stats['score'] = c.score
    stats['controversial'] = c.controversiality
    stats['prp_first'] = percent_first_pronouns(c)
    stats['prp_second'] = percent_second_pronouns(c)
    stats['prp_third'] = percent_third_pronouns(c)
    stats['sent'] = sentiment(c)
    stats['subj'] = subjectivity(c)
    stats['punc_ques'] = percent_punc_question(c)
    stats['punc_excl'] = percent_punc_exclamation(c)
    stats['punc_per'] = percent_punc_period(c)
    stats['punc'] = percent_punc(c)
    stats['profanity'] = profanity_prob(c)
    stats['hate_count'] = hate_count(c)
    stats['hedge_count'] = hedge_count(c)
    stats.update(hate_sonar(c))
    stats['is_deleted'] = ('[deleted]' == c.body or '[removed]' == c.body)

    return c.id, stats


def _blob(c: RedditComment):
    if c.blob:
        return c.blob
    else:
        blob = TextBlob(c.body)
        c.blob = blob
        return blob


def comment_languge(c: RedditComment):
    if c.body == '[deleted]':
        return 'en'
    if not c.body:
        return 'un'

    try:
        d = Detector(c.body, quiet=True)
    except:
        print(f"Failed to parse comment {c.id}")
        return 'un'

    if not d.reliable:
        return 'un'
    else:
        return d.languages[0].code


def word_count(c: RedditComment):
    if not c.body:
        return 0
    return len(_blob(c).words)


def should_nl_bail(c: RedditComment):
    if c.feats.get('word_count') is None:
        raise Exception("Must calculate word count first.")

    if c.feats['word_count'] == 0:
        return True

    if c.feats['lang'] != 'en':
        return True

    return False


def percent_first_pronouns(c: RedditComment):
    if should_nl_bail(c):
        return None

    prp = [w.lower() for w,t in _blob(c).tags if t == 'PRP']
    if len(prp) == 0:
        return 0
    prp_count = len([p for p in prp if p in eng_prp_first])
    return prp_count / len(prp)


def percent_second_pronouns(c: RedditComment):
    if should_nl_bail(c):
        return None

    prp = [w.lower() for w,t in _blob(c).tags if t == 'PRP']
    if len(prp) == 0:
        return 0
    prp_count = len([p for p in prp if p in eng_prp_second])
    return prp_count / len(prp)


def percent_third_pronouns(c: RedditComment):
    if should_nl_bail(c):
        return None

    prp = [w.lower() for w,t in _blob(c).tags if t == 'PRP']
    if len(prp) == 0:
        return 0
    prp_count = len([p for p in prp if p in eng_prp_third])
    return prp_count / len(prp)

def sentiment(c: RedditComment):
    if should_nl_bail(c):
        return None

    return _blob(c).sentiment[0]

def subjectivity(c: RedditComment):
    if should_nl_bail(c):
        return None

    return _blob(c).sentiment[1]

def percent_punc_question(c: RedditComment):
    if should_nl_bail(c):
        return None

    punc = [p for p in _blob(c).tokens if p in eng_punc]
    if len(punc) == 0:
        return 0
    ques = [p for p in punc if p == '?']
    return len(ques) / len(punc)


def percent_punc_exclamation(c: RedditComment):
    if should_nl_bail(c):
        return None

    punc = [p for p in _blob(c).tokens if p in eng_punc]
    if len(punc) == 0:
        return 0
    excl = [p for p in punc if p == '!']
    return len(excl) / len(punc)


def percent_punc_period(c: RedditComment):
    if should_nl_bail(c):
        return None

    punc = [p for p in _blob(c).tokens if p in eng_punc]
    if len(punc) == 0:
        return 0
    per = [p for p in punc if p == '.']
    return len(per) / len(punc)


def percent_punc(c: RedditComment):
    if should_nl_bail(c):
        return None
    punc = [p for p in _blob(c).tokens if p in eng_punc]
    if len(punc) == 0:
        return 0
    return len(punc)/len(_blob(c).tokens)


def profanity_prob(c: RedditComment):
    if should_nl_bail(c):
        return None
    return float(profanity([c.body])[0])

eng_hate = set(hate_dict.keys())
def hate_count(c:RedditComment):
    if should_nl_bail(c):
        return None
    count = 0
    just_text = " ".join(_blob(c).words).lower()
    for term in eng_hate:
        if term in just_text:
            count += 1
    return count

def hedge_count(c:RedditComment):
    if should_nl_bail(c):
        return None
    count = 0
    just_text = " ".join(_blob(c).words).lower()
    for term in eng_hedge:
        if term in just_text:
            count += 1
    return count

def hate_sonar(c:RedditComment):
    if should_nl_bail(c):
        return [('hate_conf',None), ('off_conf', None)]
    res = hatesonar.ping(c.body)
    d = {}
    for class_results in res['classes']:
        d[class_results['class_name']] = class_results['confidence']
    return [('hate_conf', d['hate_speech']),
            ('off_conf', d['offensive_language'])]