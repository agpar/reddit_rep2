import torch
import torch.nn as nn
#
from gat import GATFinal
from glove_embedding import word_vectors

class GloveGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(word_vectors)
        self.gat = GATFinal(embed_dim, output_size=2, K=1)

    def forward(self, inputs, offsets, adj_matrix):
        embedded = self.embedding(text, offsets)
        return self.gat(embedded, adj_matrix)
