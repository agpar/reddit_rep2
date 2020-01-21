from reddit_data.data_models import RedditComment


class RedditNode:
    def __init__(self, comment: RedditComment):
        self.comment = comment
        self.children = []
