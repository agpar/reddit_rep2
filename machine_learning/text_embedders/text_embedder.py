from abc import ABC, abstractmethod


class TextEmbedder(ABC):
    @abstractmethod
    def get_embeddings(self, trees):
        pass

    @abstractmethod
    def batch_trees(self, trees):
        pass
