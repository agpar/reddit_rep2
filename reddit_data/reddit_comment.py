class RedditComment():
    __slots__ = ('id', 'author', 'subreddit', 'score', 'body',  'parent_id', 'gilded', 'stickied', 'retrieved_on',
                 'created_utc', 'link_id', 'controversiality')

    @staticmethod
    def from_dict(comment: dict) -> 'RedditComment':
        rc = RedditComment()

        # Note: t1_ prefix is added to create "full ids" as described here: https://www.reddit.com/dev/api/
        rc.id = f't1_{comment["id"]}'
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

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id
