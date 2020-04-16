from collections import OrderedDict
import numbers


class CommentFeatures():
    def __init__(self, feats=None):
        if feats is None:
            feats = list()
        self._feats = OrderedDict(feats)

    def __getitem__(self, str_key):
        return self._feats.__getitem__(str_key)

    def __setitem__(self, str_key, value):
        return self._feats.__setitem__(str_key, value)

    def update(self, iter):
        return self._feats.update(iter)

    def get(self, key, d=None):
        return self._feats.get(key, d)

    def to_vector(self):
        items = self._feats.items()
        return list(float(v) for k, v in items if isinstance(v, numbers.Number))

    def vector_labels(self):
        items = self._feats.items()
        return list(k for k, v in items if isinstance(v, numbers.Number))

    def to_labelled_vector(self):
        return list(zip(self.vector_labels(), self.to_vector()))