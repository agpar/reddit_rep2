from typing import Collection
from collections import namedtuple

import psycopg2

from reddit_data.data_sources.data_source import DataSource
from reddit_data.reddit_comment import RedditComment

PSQLServerDef = namedtuple('PSQLServerDef', ['host', 'database', 'user', 'password'])


class PSQLDataSource(DataSource):

    def __init__(self, server_def: PSQLServerDef):
        self._connection = psycopg2.connect(**server_def._asdict())

    def __del__(self):
        self._connection.close()

    def get_comment(self, comment_id: str) -> RedditComment:
        query = f"""
            SELECT * FROM comments
            WHERE id = '{id}'
        """
        cursor = self._connection.cursor()
        cursor.execute(query)
        #TODO: Does this handle empty results?
        return self._row_to_comment(cursor, cursor.fetchone())

    def get_children(self, id: str) -> Collection[RedditComment]:
        query = f"""
            SELECT * FROM comments
            WHERE parent_id = '{id}';
        """
        cursor = self._connection.cursor()
        cursor.execute(query)

        #TODO: Does this handle empty results?
        return self._rows_to_comments(cursor)

    def _rows_to_comments(self, cursor):
        comments = set()
        for row in cursor.fetchall():
            comments.add(self._row_to_comment(cursor, row))
        return comments

    #TODO maybe pass in a row and a row description?
    def _row_to_comment(self, cursor, row):
        return RedditComment.from_dict(self._row_to_dict(cursor, row))

    def _row_to_dict(self, cursor, row):
        d = {}
        for i, col in enumerate(cursor.description):
            d[col[0]] = row[i]
        return d
