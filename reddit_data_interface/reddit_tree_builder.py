import math
from reddit_data_interface.data_models import RedditComment
from reddit_data_interface.data_models.reddit_node import RedditNode
from reddit_data_interface.data_sources.data_source import DataSource


class RedditTreeBuilder:
    def __init__(self, data_source: DataSource):
        self.data = data_source

    def get_tree_rooted_at(self, parent_comment: RedditComment, depth=math.inf) -> RedditNode:
        return self._assign_indexes(self._build_tree(parent_comment, depth))

    def _build_tree(self, parent_comment, depth):
        if parent_comment is None or depth <= 0:
            return None

        parent_node = RedditNode(parent_comment)
        child_comments = self.data.get_children(parent_comment.id)
        for child_comment in sorted(child_comments):
            child_node = self._build_tree(child_comment, depth - 1)
            if child_node is not None:
                parent_node.children.append(child_node)

        return parent_node

    def _assign_indexes(self, tree):
        index = 0
        for node in tree:
            node.index = index
            index += 1
        return tree



