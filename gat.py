import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


class GAT(nn.Module):

    def __init__(self, input_size, output_size, K, adj):
        super(GAT, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.K = K
        self.adj = adj # adjacency matrix. adj[i,:] is neighbors of ith node

        """
        Registering W and a as `Parameters` means that forward operations on them will be tracked
        so that a backwards (optimizing) pass can be derived. See the function `train()` at the
        bottom of this module for how easy pytorch makes this.
        """
        self.W = nn.Parameter(torch.rand(K, output_size, input_size))
        self.a = nn.Parameter(torch.rand(K, 1, 2*output_size))

    def forward(self, X):
        """Expecting to be passed the entire graph in X."""
        results = torch.empty(0, self.output_size * self.K)
        for i, x in enumerate(X):
            multi_head_results = torch.empty(1, 0)
            for k in range(self.K):
                one_head_result = self.compute_embedding(X, i, k)
                multi_head_results = torch.cat((multi_head_results, one_head_result), dim=1)
            results = torch.cat((results, multi_head_results), dim=0)
        return results

    def compute_embedding(self, X, i, k):
        """Compute the k'th embedding of the i'th node"""
        a = self.a[k]
        W = self.W[k]
        x = X[i]

        # Construct a matrix where each row is the mapped result of x and
        # one of its neighbours concated.
        neighbours = X[self.sample_neighborhood(i)].view(-1, self.input_size)
        mapped_neighbours = neighbours.mm(W.t())
        mapped_x_repeat = x.view(1, -1).mm(W.t()).repeat(neighbours.shape[0], 1)
        neighbour_cat_x = torch.cat((mapped_neighbours, mapped_x_repeat), dim=1)

        # Compute the weights given to each of the neighbours based on
        # the attention function.
        attention = F.softmax(F.leaky_relu(neighbour_cat_x.mm(a.t())), dim=0)

        # Combine mapped neighbours based on attention weighting.
        return (mapped_neighbours.t().mm(attention)).t()

    def sample_neighborhood(self, i):
        """Currently return all neighbours"""
        return (self.adj[i,:] != 0).nonzero()


# A simple function to train a net.
def train(net, X, y, lr=0.00000001, iters=100):
    optimizer = optim.SGD(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for i in range(iters):
        optimizer.zero_grad()
        output = net.forward(X)
        loss = criterion(output, y)
        print(loss)
        loss.backward()
        optimizer.step()
