import math
import pickle
#
import torch
import torch.nn as nn
import torch.optim as optim
#
from gat import GATFinal
from glove_embedding import EMBED_DIM, word_vectors

class GloveGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(torch.tensor(word_vectors))
        self.gat = GATFinal(EMBED_DIM, output_size=1, K=1)

    def forward(self, inputs, offsets, adj_matrix):
        embedded = self.embedding(inputs, offsets).float()
        return self.gat(embedded, adj_matrix)

def create_train_and_test(data):
    pass

def predict(model, data):
    pass

def train(model, inputs, offsets, adj_matrix, cls, lr=0.0001, iters=100):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    tenth_iter = math.floor(iters / 10)
    for i in range(iters):
        optimizer.zero_grad()
        output = model.forward(inputs, offsets, adj_matrix)
        loss = criterion(output, cls)
        if (i % tenth_iter == 0):
            print(loss)
        loss.backward()
        optimizer.step()

with open("sample_data.pkl", 'rb') as input:
    model_data = pickle.load(input)

model = GloveGAT()

adj_matrix = model_data["adj_matrix"]
embedding = model_data["embedding"]
offsets = model_data["offsets"]
outputs = model_data["outputs"]

train(model, embedding, offsets, adj_matrix, outputs)
