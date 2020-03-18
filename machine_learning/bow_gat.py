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
from gat import GAT, GATFinal
from train_utils import assert_vars_change, create_train_and_test

class BagOfWordsGAT(nn.Module):
    def __init__(self, vectorizer):
        super().__init__()
        self.vectorizer = vectorizer
        self.gat1 = GAT(len(vectorizer.get_feature_names()), output_size=8, K=8)
        self.gat2 = GATFinal(64, output_size=2, K=1)

    def forward(self, inputs, adj_matrix):
        hidden = self.gat1(inputs, adj_matrix)
        return self.gat2(hidden, adj_matrix)

with open("sample_bow_data.pkl", 'rb') as input:
    model_data = pickle.load(input)

tree_data = model_data["tree_data"]
vectorizer = model_data["vectorizer"]

model = BagOfWordsGAT(vectorizer)
optimizer = optim.SGD(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()
sub_train, sub_valid = create_train_and_test(tree_data)

def train_step(model, criterion, optimizer, batch):
    adj_matrix = batch["adj_matrix"]
    inputs = batch["bag_of_words"]
    targets = batch["outputs"]

    optimizer.zero_grad()
    outputs = model(inputs, adj_matrix)
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
    inputs = batch["bag_of_words"]
    y = batch["outputs"].argmax(1)
    with torch.no_grad():
        y_pred = model(inputs, adj_matrix).argmax(1)
    return y_pred, y

def test_variables_change(model, batch):
    model.train()
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
        exit()

def get_model_params(model):
    params = [ np for np in model.named_parameters() if np[1].requires_grad ]
    return params

test_variables_change(model, tree_data[:3])

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

# trainer.run(sub_train, max_epochs=5)


