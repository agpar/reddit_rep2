from reddit_data import RedditTreeBuilder, JsonDataSource

# 1. Get json formatted reddit data from https://files.pushshift.io/reddit/comments/
# 2. Use filter_subreddits.py to produce a single file containing only the subreddits you care about.
# 3. Look at example below on how to load a filtered json file and iterate over trees of comments.

path_to_json_data = "reddit_data/RC_2006-12" # for example, use comments from 2006
jds = JsonDataSource(path_to_json_data)
rt = RedditTreeBuilder(jds)

all_roots = list(jds.get_roots())
all_trees = [rt.get_tree_rooted_at(c) for c in all_roots]
