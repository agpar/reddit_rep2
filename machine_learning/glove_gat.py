import math
import pickle
#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import random_split
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Precision, Recall
#
from gat import GATFinal
from glove_embedding import EMBED_DIM, word_vectors

class GloveGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(torch.tensor(word_vectors))
        self.gat = GATFinal(EMBED_DIM, output_size=2, K=1)

    def forward(self, inputs, offsets, adj_matrix):
        embedded = self.embedding(inputs, offsets).float()
        return self.gat(embedded, adj_matrix)

def create_train_and_test(data, split_percent=0.9):
    train_len = int(len(model_data) * split_percent)
    sub_train, sub_valid = random_split(model_data, [train_len, len(model_data) - train_len])
    return sub_train, sub_valid

def predict(model, data):
    preds = []
    for tree_data in data:
        output = model(
            tree_data["embedding"],
            tree_data["offsets"],
            tree_data["adj_matrix"]
        )
        pred = output.argmax(1)
        preds.append(pred)
    return preds

def get_percent_positive(data):
    num_nodes = sum([len(tree["outputs"]) for tree in data])

    # This relies on the fact that there are only two classes and will have
    # the values 0 for positive and 1 for negative
    num_negative = [tree["outputs"].argmax(1).sum().item() for tree in data]
    total_negative = sum(num_negative)

    return 1 - (total_negative / num_nodes)

def train(model, data, lr=0.01):
    # Metrics
    train_loss = 0
    train_acc = 0
    num_nodes = sum([len(tree["outputs"]) for tree in data])

    # model setup
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    tenth_iter = math.floor(len(data) / 10)
    for i, tree_data in enumerate(data):
        adj_matrix = tree_data["adj_matrix"]
        inputs = tree_data["embedding"]
        offsets = tree_data["offsets"]
        cls = tree_data["outputs"]

        optimizer.zero_grad()
        output = model.forward(inputs, offsets, adj_matrix)
        loss = criterion(output, cls)
        train_loss += loss.item()
        if (i % tenth_iter == 0):
            print(loss)
        loss.backward()
        optimizer.step()

        num_correct = (output.argmax(1) == cls.argmax(1)).sum().item()
        train_acc += num_correct

    overall_loss = train_loss / num_nodes
    overall_acc = train_acc / num_nodes
    print(f"loss: {overall_loss: .4f}")
    print(f"acc: {overall_acc: .4f}")


with open("sample_data.pkl", 'rb') as input:
    model_data = pickle.load(input)

model = GloveGAT()
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
sub_train, sub_valid = create_train_and_test(model_data)

def update_model(engine, batch):
    adj_matrix = batch["adj_matrix"]
    inputs = batch["embedding"]
    offsets = batch["offsets"]
    targets = batch["outputs"]

    optimizer.zero_grad()
    outputs = model(inputs, offsets, adj_matrix)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def predict_on_batch(engine, batch):
    model.eval()
    adj_matrix = batch["adj_matrix"]
    inputs = batch["embedding"]
    offsets = batch["offsets"]
    y = batch["outputs"].argmax(1)
    with torch.no_grad():
        y_pred = model(inputs, offsets, adj_matrix).argmax(1)
    return y_pred, y

trainer = Engine(update_model)
evaluator = Engine(predict_on_batch)
Accuracy().attach(evaluator, "accuracy")
Precision().attach(evaluator, "precision")
Recall().attach(evaluator, "recall")

@trainer.on(Events.ITERATION_COMPLETED(every=100))
def log_training(engine):
    batch_loss = engine.state.output
    lr = optimizer.param_groups[0]['lr']
    epoch = engine.state.epoch
    max_epochs = engine.state.max_epochs
    iteration = engine.state.iteration
    print(f"Epoch {epoch}/{max_epochs} : {iteration} - batch loss: {batch_loss: .4f}, lr: {lr}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(sub_valid)
    metrics = evaluator.state.metrics
    epoch = trainer.state.epoch

    accuracy = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    print(f"Validation Results - Epoch: {epoch} "\
          f"Avg accuracy: {accuracy:.2f} Avg precision: {precision:.2f} Avg recall: {recall:.2f}")

trainer.run(sub_train, max_epochs=5)

