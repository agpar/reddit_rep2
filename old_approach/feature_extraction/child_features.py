"""
Features averaged over the direct children of a comment.
"""
from old_approach.feature_extraction.comment_features import CommentFeatures
from reddit_data_interface.data_models.reddit_node import RedditNode

import numpy as np


def compute_child_features(n: RedditNode):
    n.ch_feats = CommentFeatures()
    stats = n.ch_feats

    stats.update(multi(n, lambda x: x.comment.score, 'score'))
    stats.update(multi(n, lambda x: x.comment.feats['prp_first'], 'prp_first'))
    stats.update(multi(n, lambda x: x.comment.feats['prp_second'], 'prp_second'))
    stats.update(multi(n, lambda x: x.comment.feats['prp_third'], 'prp_third'))
    stats.update(multi(n, lambda x: x.comment.feats['sent'], 'sent'))
    stats.update(multi(n, lambda x: x.comment.feats['subj'], 'subj'))
    stats.update(multi(n, lambda x: x.comment.feats['punc_ques'], 'punc_ques'))
    stats.update(multi(n, lambda x: x.comment.feats['punc_excl'], 'punc_excl'))
    stats.update(multi(n, lambda x: x.comment.feats['punc_per'], 'punc_per'))
    stats.update(multi(n, lambda x: x.comment.feats['punc'], 'punc'))
    stats.update(multi(n, lambda x: x.comment.feats['profanity'], 'profanity'))
    stats.update(multi(n, lambda x: x.comment.feats['hate_count'], 'hate_count'))
    stats.update(multi(n, lambda x: x.comment.feats['hedge_count'], 'hedge_count'))
    stats.update(multi(n, lambda x: x.comment.feats['hate_conf'], 'hate_conf'))
    stats.update(multi(n, lambda x: x.comment.feats['off_conf'], 'off_conf'))

    stats['child_score_disag'] = child_disagreement(n)
    stats['child_contro'] = child_contro(n)
    stats['child_deleted'] = child_deleted(n)
    stats['child_count'] = len(n.children)


def _avg(node, selector):
    data = [selector(c) for c in node.children]
    data = [d for d in data if d is not None]
    if not data:
        return None

    return np.mean(data)


def _std(node, selector):
    data = [selector(c) for c in node.children]
    data = [d for d in data if d is not None]
    if not data:
        return None

    return np.std(data)


def _min(node, selector):
    data = [selector(c) for c in node.children]
    data = [d for d in data if d is not None]
    if not data:
        return None

    return min(data)


def _max(node, selector):
    data = [selector(c) for c in node.children]
    data = [d for d in data if d is not None]
    if not data:
        return None

    return max(data)


def _median(node, selector):
    data = [selector(c) for c in node.children]
    data = [d for d in data if d is not None]
    if not data:
        return None

    data.sort()
    return data[int(len(data)/2)]


def multi(node, selector, label):
    agg_funcs = [_avg, _std, _min, _max, _median]
    agg_labs = ['avg', 'std', 'min', 'max', 'med']
    features = []
    for fn, fn_name in zip(agg_funcs, agg_labs):
        feat_label = f"child_{fn_name}_{label}"
        features.append((feat_label, fn(node,selector)))
    return features


def child_disagreement(node):
    if len(node.children) == 0:
        return None
    if len(node.children) < 2:
        return 0.0

    child_scores = [c.comment.score for c in node.children]
    neg_scores = [s for s in child_scores if s < 1]
    pos_scores = [s for s in child_scores if s > 1]
    if len(pos_scores) == 0:
        return 1.0

    return len(neg_scores)/len(pos_scores)


def child_contro(node):
    if len(node.children) == 0:
        return None
    child_contro = [c.comment.feats['controversial'] for c in node.children]
    pos = [s for s in child_contro if s]
    return len(pos) / len(child_contro)


def child_deleted(node):
    if len(node.children) == 0:
        return None
    child_deleted = [c.comment.feats['is_deleted'] for c in node.children]
    pos = [s for s in child_deleted if s]
    return len(pos) / len(child_deleted)