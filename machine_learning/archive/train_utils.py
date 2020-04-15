import torch
from torch.utils.data.dataset import random_split

def create_train_and_test(model_data, split_percent=0.9):
    train_len = int(len(model_data) * split_percent)
    sub_train, sub_valid = random_split(model_data, [train_len, len(model_data) - train_len])
    return sub_train, sub_valid


def get_percent_positive(data):
    num_nodes = sum([len(tree["outputs"]) for tree in data])

    # This relies on the fact that there are only two classes and will have
    # the values 0 for positive and 1 for negative
    num_negative = [tree["outputs"].argmax(1).sum().item() for tree in data]
    total_negative = sum(num_negative)

    return 1 - (total_negative / num_nodes)


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

