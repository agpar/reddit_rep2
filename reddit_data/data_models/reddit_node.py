from collections import deque

from reddit_data.data_models import RedditComment


class RedditNode:
    def __init__(self, comment: RedditComment):
        self.comment = comment
        self.index = None
        self.children = []

    def __iter__(self):
        queue = deque()
        queue.append(self)
        while queue:
            node = queue.popleft()
            yield node
            for child in node.children:
                queue.append(child)
