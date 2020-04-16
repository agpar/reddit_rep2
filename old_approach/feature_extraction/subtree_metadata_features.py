"""Feature extraction tools that operate on a Comment (tree) structure"""

import math
import numpy as np

from old_approach.feature_extraction.comment_features import CommentFeatures
from old_approach.feature_extraction.subtree_features import SubtreeFeatures
from reddit_data_interface.data_models.reddit_node import RedditNode


def compute_subtree_metadata_features(n: RedditNode, stf : SubtreeFeatures):
    n.st_feats = CommentFeatures()
    stats = n.st_feats

    stats['desc_size'] = tree_size(n)
    stats['desc_depth'] = tree_depth(n)
    stats['desc_contro'] = percent_controversial(n, stf)
    stats['desc_score_disag'] = disagreement(n, stf)

    stats.update(multi(stf, lambda x: x.scores, 'score'))
    stats.update(multi(stf, lambda x: x.prp_first, 'prp_first'))
    stats.update(multi(stf, lambda x: x.prp_second, 'prp_second'))
    stats.update(multi(stf, lambda x: x.prp_third, 'prp_third'))
    stats.update(multi(stf, lambda x: x.sent, 'sent'))
    stats.update(multi(stf, lambda x: x.subj, 'subj'))

    stats.update(multi(stf, lambda x: x.punc_ques, 'punc_ques'))
    stats.update(multi(stf, lambda x: x.punc_excl, 'punc_excl'))
    stats.update(multi(stf, lambda x: x.punc_per, 'punc_per'))

    stats.update(multi(stf, lambda x: x.profanity, 'profanity'))
    stats.update(multi(stf, lambda x: x.hate_count, 'hate_count'))
    stats.update(multi(stf, lambda x: x.hedge_count, 'hedge_count'))
    stats.update(multi(stf, lambda x: x.hate_conf, 'hate_conf'))
    stats.update(multi(stf, lambda x: x.off_conf, 'off_conf'))

    # all deleted and contro
    stats['desc_contro'] = desc_contro(stf)
    stats['desc_deleted'] = desc_deleted(stf)


def _avg(stf, selector):
    data = selector(stf)
    data = [d for d in data if d is not None]
    if not data:
        return None

    return np.mean(data)


def _std(stf, selector):
    data = selector(stf)
    data = [d for d in data if d is not None]
    if not data:
        return None

    return np.std(data)


def _min(stf, selector):
    data = selector(stf)
    data = [d for d in data if d is not None]
    if not data:
        return None

    return min(data)


def _max(stf, selector):
    data = selector(stf)
    data = [d for d in data if d is not None]
    if not data:
        return None

    return max(data)


def _median(stf, selector):
    data = selector(stf)
    data = [d for d in data if d is not None]
    if not data:
        return None

    data.sort()
    return data[int(len(data)/2)]


def multi(stf, selector, label):
    agg_funcs = [_avg, _std, _min, _max, _median]
    agg_labs = ['avg', 'std', 'min', 'max', 'med']
    features = []
    for fn, fn_name in zip(agg_funcs, agg_labs):
        feat_label = f"desc_{fn_name}_{label}"
        features.append((feat_label, fn(stf,selector)))
    return features


def tree_size(node: RedditNode):
    if len(node.children) > 0:
        return sum([(1 + c.st_feats['desc_size']) for c in node.children])
    else:
        return 0


def tree_depth(node):
    if len(node.children) > 0:
        return 1 + max([c.st_feats['desc_depth'] for c in node.children])
    else:
        return 0


def percent_controversial(node: RedditNode, subtree_features):
    if node.st_feats['desc_size'] == 0:
        return None
    return subtree_features.controversial_count / node.st_feats['desc_size']


def disagreement(node, stf):
    if len(stf.scores) == 0:
        return None
    if len(stf.scores) < 2:
        return 0.0

    neg_scores = [s for s in stf.scores if s < 1]
    pos_scores = [s for s in stf.scores if s > 1]
    if len(pos_scores) == 0:
        return 1.0

    return len(neg_scores)/len(pos_scores)


def desc_contro(stf):
    if len(stf.comments) == 0:
        return None
    return stf.controversial_count / len(stf.comments)


def desc_deleted(stf):
    if len(stf.comments) == 0:
        return None
    return stf.deleted_count / len(stf.comments)
