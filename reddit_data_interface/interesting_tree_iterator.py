import math

from reddit_data_interface import RedditTreeBuilder, DataSource


class InterestingTreeIterator:
    def __init__(self, data_source: DataSource, tree_builder: RedditTreeBuilder):
        self.tree_builder = tree_builder
        self.data_source = data_source

    def __iter__(self):
        return self.iter_interesting_trees()

    def iter_interesting_trees(self, depth=math.inf):
        if depth < 3:
            raise ValueError("depth must be >= 3 for the current definition of 'interesting'")

        for parent in self.data_source.iter_parents():
            parent_tree = self.tree_builder.get_tree_rooted_at(parent, depth=depth)
            if self._is_interesting(parent_tree):
                yield parent_tree

    def _is_interesting(self, tree):
        return len(tree.children) >= 2 and any(c.children for c in tree.children)