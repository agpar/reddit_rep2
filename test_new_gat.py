import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import random_split
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Precision, Recall
#
from pyGAT.models import GAT
from text_preprocessing import stemming_tokenizer, get_parent_indices

class FilterGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(FilterGAT, self).__init__()
        self.gat = GAT(nfeat, nhid, nclass, dropout, alpha, nheads)

    def forward(self, x, adj):
        x = self.gat(x, adj)
        # Filter out leaf nodes
        parent_indices = get_parent_indices(adj)
        return x[parent_indices]

def create_train_and_test(model_data, split_percent=0.9):
    train_len = int(len(model_data) * split_percent)
    sub_train, sub_valid = random_split(model_data, [train_len, len(model_data) - train_len])
    return sub_train, sub_valid

# Modified version from torchtest to accomodate multiple input variables
def assert_vars_change(model, loss_fn, optim, batch, train_step, device="cpu"):
    params = [ np for np in model.named_parameters() if np[1].requires_grad ]
    # take a copy
    initial_params = [ (name, p.clone()) for (name, p) in params ]

    # run a training step
    if type(batch) is list:
        for data in batch:
            train_step(model, loss_fn, optim, data)
    else:
        train_step(model, loss_fn, optim, batch)

    unchanged_names = []
    # check if variables have changed
    for (_, p0), (name, p1) in zip(initial_params, params):
        if (torch.equal(p0.to(device), p1.to(device))):
            unchanged_names.append(name)

    if (len(unchanged_names) > 0):
        raise Exception(f"{unchanged_names} did not change!")

def test_variables_change(model, batch):
    model.train()
    try:
        assert_vars_change(
            model=model,
            loss_fn=F.nll_loss,
            optim=optimizer,
            batch=batch,
            train_step=train_step,
            device="cpu"
        )
        print("SUCCESS: variables changed")
    except Exception as e:
        print("FAILED: ", e)

with open("./machine_learning/sample_bow_data.pkl", 'rb') as input:
    model_data = pickle.load(input)

tree_data = model_data["tree_data"]
vectorizer = model_data["vectorizer"]

model = FilterGAT(nfeat=len(vectorizer.get_feature_names()),
                nhid=8,
                nclass=2,
                dropout=0.6,
                nheads=8,
                alpha=0.2)
optimizer = optim.Adam(model.parameters(),
                       lr=0.005,
                       weight_decay=5e-4)
criterion=F.nll_loss

def train_step(model, criterion, optimizer, batch):
    adj_matrix = batch["adj_matrix"].to_dense()
    inputs = batch["bag_of_words"]
    targets = batch["outputs"].argmax(dim=1)

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
    adj_matrix = batch["adj_matrix"].to_dense()
    inputs = batch["bag_of_words"]
    y = batch["outputs"].argmax(dim=1)
    with torch.no_grad():
        y_pred = model(inputs, adj_matrix).argmax(dim=1)
    return y_pred, y



test_variables_change(model, tree_data[0])

sub_train, sub_valid = create_train_and_test(tree_data)

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
