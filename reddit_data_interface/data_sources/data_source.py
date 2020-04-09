from abc import ABC, abstractmethod
from typing import Collection, Iterable

from reddit_data_interface.data_models import RedditComment


class DataSource(ABC):

    @abstractmethod
    def get_comment(self, comment_id: str) -> RedditComment:
        pass

    @abstractmethod
    def get_children(self, parent_id: str) -> Collection[RedditComment]:
        pass

    @abstractmethod
    def get_roots(self) -> Iterable[RedditComment]:
        pass
