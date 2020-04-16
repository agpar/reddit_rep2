from old_approach.feature_extraction.child_features import compute_child_features
from old_approach.feature_extraction.comment_features import CommentFeatures
from old_approach.feature_extraction.nl_features import compute_nl_features
from old_approach.feature_extraction.subtree_features import SubtreeFeatures
from old_approach.feature_extraction.subtree_metadata_features import compute_subtree_metadata_features
from reddit_data_interface.data_models.reddit_node import RedditNode


def tree_to_vects(node):
    annotate_tree(node)
    feats = CommentFeatures()
    feats.update(node.ch_feats._feats)
    feats.update(node.st_feats._feats)

    labels = CommentFeatures()
    labels.update(node.comment.feats._feats)
    return feats, labels


def annotate_tree(node) -> (RedditNode, SubtreeFeatures):
    subtree_features = []

    for n in node.children:
        subtree, features = annotate_tree(n)
        subtree_features.append(features)

    combined_features = SubtreeFeatures.combine(subtree_features)
    _compute_features(node, combined_features)
    combined_features.update(node)
    return node, combined_features


def _compute_features(n: RedditNode, stf: SubtreeFeatures):
    """Computes an associates aggregate features of this subtree"""

    # subtree features
    compute_subtree_metadata_features(n, stf)

    # Children stats
    compute_child_features(n)

    # Natural language stats
    #compute_nl_features(n)