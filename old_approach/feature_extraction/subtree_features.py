class SubtreeFeatures:
    """Stores comments of subtree to help with recursive feautre building."""

    def __init__(self):
        self.comments = []

    @staticmethod
    def combine(subtree_features_list):
        """Returns a new instance with properties of all in given list."""
        combined = SubtreeFeatures()

        for stf in subtree_features_list:
            combined.comments.extend(stf.comments)

        return combined

    def update(self, comment):
        """Updates this instance with the features from the given comment."""
        self.comments.append(comment)

    @property
    def scores(self):
        return [c.comment.score for c in self.comments]

    @property
    def controversial_count(self):
        return sum(1 for c in self.comments if c.comment.controversiality)

    @property
    def prp_first(self):
        return [c.comment.feats['prp_first'] for c in self.comments]

    @property
    def prp_second(self):
        return [c.comment.feats['prp_second'] for c in self.comments]

    @property
    def prp_third(self):
        return [c.comment.feats['prp_third'] for c in self.comments]

    @property
    def sent(self):
        return [c.comment.feats['sent'] for c in self.comments]

    @property
    def subj(self):
        return [c.comment.feats['subj'] for c in self.comments]

    @property
    def punc_ques(self):
        return [c.comment.feats['punc_ques'] for c in self.comments]

    @property
    def punc_excl(self):
        return [c.comment.feats['punc_excl'] for c in self.comments]

    @property
    def punc_per(self):
        return [c.comment.feats['punc_per'] for c in self.comments]

    @property
    def profanity(self):
        return [c.comment.feats['profanity'] for c in self.comments]

    @property
    def hate_count(self):
        return [c.comment.feats['hate_count'] for c in self.comments]

    @property
    def hedge_count(self):
        return [c.comment.feats['hedge_count'] for c in self.comments]

    @property
    def hate_conf(self):
        return [c.comment.feats['hate_conf'] for c in self.comments]

    @property
    def off_conf(self):
        return [c.comment.feats['off_conf'] for c in self.comments]

    @property
    def deleted_count(self):
        return sum(1 for c in self.comments if c.comment.feats['is_deleted'])
