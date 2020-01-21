#!/usr/bin/bash

import argparse
import json

from data_models import RedditComment

FIELDS = (
    ('id', int),
    ('author', str),
    ('subreddit', str),
    ('score', int),
    ('body', str),
    ('parent_id', str),
    ('gilded', int),
    ('stickied', bool),
    ('retrieved_on', int),
    ('created_utc', int),
    ('link_id', str),
    ('controversiality', int),
)


def print_setup():
    print("""CREATE TABLE comments(
        id VARCHAR(20) PRIMARY KEY,
        author VARCHAR(100),
        subreddit VARCHAR(20),
        score INT,
        body TEXT,
        parent_id VARCHAR(20),
        gilded INT,
        stickied boolean,
        retrieved_on INT,
        created_utc INT,
        link_id VARCHAR(20),
        controversiality INT);
    """)


def print_indexes():
    print("CREATE INDEX parents on COMMENTS(parent_id);")


def convert_file_to_sql(file):
    with open(file, 'r') as f:
        for line in f:
            print(f"INSERT INTO comments ({json_line_to_sql(line)});")


def json_line_to_sql(json_line):
    json_obj = json.loads(json_line)
    rc = RedditComment.from_dict(json_obj)
    values = []
    for field, field_type in FIELDS:
        if field_type == int:
            values.append(str(rc.__getattribute__(field)))
        if field_type == str:
            sanitized = rc.__getattribute__(field).replace('"', '\\"')
            values.append(f'\"{sanitized}\"')
        if field_type == bool:
            values.append('TRUE' if rc.__getattribute__(field) else 'FALSE')
    return ",".join(str(s) for s in values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter out lines that are not from subreddits of interest.")
    parser.add_argument("--include-setup", default=False, action='store_true',
                        help="Use this flag to include setup of databases and indexes")
    parser.add_argument("json_files", type=str, nargs='+',
                        help="One or more json files to convert to SQL.")
    args = parser.parse_args()

    json_files = args.json_files
    include_setup = args.include_setup

    if include_setup:
        print_setup()

    for file in json_files:
        convert_file_to_sql(file)

    if include_setup:
        print_indexes()
