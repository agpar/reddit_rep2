import math
import pickle
#
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Precision, Recall
#
from gat import GATFinal
from glove_embedding import EMBED_DIM, word_vectors
from train_utils import assert_vars_change, create_train_and_test

class GloveGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(torch.tensor(word_vectors))
        self.gat = GATFinal(EMBED_DIM, output_size=2, K=1)

    def forward(self, inputs, offsets, adj_matrix):
        embedded = self.embedding(inputs, offsets).float()
        return self.gat(embedded, adj_matrix)

with open("sample_data.pkl", 'rb') as input:
    model_data = pickle.load(input)

model = GloveGAT()
optimizer = optim.SGD(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()
sub_train, sub_valid = create_train_and_test(model_data)

def train_step(model, criterion, optimizer, batch):
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

def update_model(engine, batch):
    model.train()
    return train_step(model, criterion, optimizer, batch)

def predict_on_batch(engine, batch):
    model.eval()
    adj_matrix = batch["adj_matrix"]
    inputs = batch["embedding"]
    offsets = batch["offsets"]
    y = batch["outputs"].argmax(1)
    with torch.no_grad():
        y_pred = model(inputs, offsets, adj_matrix).argmax(1)
    return y_pred, y


def test_variables_change(model, batch):
    try:
        assert_vars_change(
            model=model,
            loss_fn=criterion,
            optim=optimizer,
            batch=batch,
            train_step=train_step,
            device="cpu"
        )
        print("SUCCESS: variables changed")
    except Exception as e:
        print("FAILED: ", e)
        exit(1)

test_variables_change(model, model_data[0])

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

trainer.run(sub_train, max_epochs=10)

