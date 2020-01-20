import json
from collections import defaultdict
from typing import Collection

from reddit_data.data_sources.data_source import DataSource
from reddit_data.reddit_comment import RedditComment


class JsonDataSource(DataSource):
    """
    Load an entire file of Reddit comments into memory and creates indexes with dicts.
    """

    def __init__(self, data_file: str):
        self.comments_by_id = {}
        self.comments_by_parent_id = defaultdict(set)
        self._load_data(data_file)

    def _load_data(self, data_file: str):
        with open(data_file, 'r') as f:
            for line in f:
                rc = RedditComment.from_dict(json.loads(line))
                self.comments_by_id[rc.id] = rc
                self.comments_by_parent_id[rc.parent_id].add(rc)

    def get_comment(self, comment_id: str) -> RedditComment:
        return self.comments_by_id.get(comment_id, None)

    def get_children(self, parent_id: str) -> Collection[RedditComment]:
        return self.comments_by_parent_id.get(parent_id, set())
