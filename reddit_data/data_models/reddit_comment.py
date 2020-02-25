import re

from reddit_data.data_parsing_exception import DataParsingException


class RedditComment():
    __slots__ = ('id', 'author', 'subreddit', 'score', 'body',  'parent_id', 'gilded', 'stickied', 'retrieved_on',
                 'created_utc', 'link_id', 'controversiality')

    COMMENT_ID_PREFIX = 't1_'
    LINK_ID_PREFIX = 't3_'
    PREFIX_REGEX = re.compile('^t.*_')

    @staticmethod
    def from_dict(comment: dict) -> 'RedditComment':
        rc = RedditComment()
        rc.id = RedditComment._full_id(comment['id'])
        rc.author = comment['author']
        rc.subreddit = comment['subreddit']
        rc.score = comment['score']
        rc.body = comment['body']
        rc.parent_id = comment['parent_id']
        rc.stickied = comment['stickied']
        rc.retrieved_on = comment['retrieved_on']
        rc.created_utc = comment['created_utc']
        rc.gilded = comment['gilded']
        rc.link_id = comment['link_id']
        rc.controversiality = comment['controversiality']
        return rc

    @staticmethod
    def _full_id(comment_id):
        if comment_id.startswith(RedditComment.COMMENT_ID_PREFIX):
            return comment_id
        else:
            if RedditComment.PREFIX_REGEX.match(comment_id):
                raise DataParsingException(f"Comment id has unknown prefix: {comment_id}")
            else:
                return RedditComment.COMMENT_ID_PREFIX + comment_id

    def is_root(self):
        """True when this comment was directly replying to a link."""
        return self.parent_id.startswith(RedditComment.LINK_ID_PREFIX)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id
