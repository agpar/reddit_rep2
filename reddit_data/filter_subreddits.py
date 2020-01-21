#!/usr/bin/python3

import argparse
import os
import sys
import re

subreddit_regex = re.compile('\"subreddit\":\"(\w+)\"')


def find_subreddit(line):
    match = subreddit_regex.search(line)
    if not match:
        return None
    else:
        return match.group(1).lower()


def filter_file(file_path, valid_subreddits):
    with open(file_path, 'r') as f:
        for line in f:
            if find_subreddit(line) in valid_subreddits:
                print(line, end="")


def only_existing_files(json_files):
    existing_files = []
    for json_file in json_files:
        if not os.path.exists(json_file):
            print(f"Error: file does not exist. Skipping: {json_file}", file=sys.stderr)
        else:
            existing_files.append(json_file)
    return existing_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter out lines that are not from subreddits of interest.")
    parser.add_argument("subreddits", type=str, nargs=1,
                        help="A comma separated list of subreddits of interest, e.g. 'politics,askreddit,movies'")
    parser.add_argument("json_files", type=str, nargs='+',
                        help="One or more json files to filter.")
    args = parser.parse_args()

    json_files = args.json_files
    valid_subreddits = {s.lower() for s in args.subreddits[0].split(',')}
    for json_file in only_existing_files(json_files):
        filter_file(json_file, valid_subreddits)


