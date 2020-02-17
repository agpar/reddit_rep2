import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtest import assert_vars_change
from gat import train, GAT, GATFinal

input_size = 2 # number of input features per node
output_size = 1 # number of output features per node
k = 1 # number of attention mechanisms

# adjacency matrix
edges = torch.LongTensor([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]])
ones = torch.ones(edges.size(0))
adj_matrix = torch.sparse.IntTensor(edges.t(), ones)

# input values
nodes = torch.FloatTensor([
    [1, 2],
    [2, 2],
    [-1, 1],
    [3, 3]
])
# Outputs are calculated with the following linear equation
# with a node [a, b], a node is assigned the value 2a + b
# The output is the avg of all neighbour values
outputs = F.softmax(torch.FloatTensor([6, 1.5, 7.5, -1]), dim=0)

net = GATFinal(input_size, output_size, k, adj_matrix)

# Sanity check to make sure parameters are being trained
def test_variables_change(model, inputs, outputs):
    try:
        assert_vars_change(
            model=model,
            loss_fn=nn.MSELoss(),
            optim=torch.optim.SGD(net.parameters(), lr=0.0001),
            batch=[inputs, outputs],
            device="cpu"
        )
        print("SUCCESS: variables changed")
    except Exception as e:
        print("FAILED: ", e)


def test_compute_embedding():
    # Setup a simple 3 node graph with arbitrary values
    test_edges = torch.LongTensor([[1, 0], [1, 2]])
    test_adj = torch.sparse.IntTensor(test_edges.t(), torch.ones(test_edges.size(0)))
    test_nodes = torch.FloatTensor([
        [1, 1],
        [0, 1],
        [2, 3]
    ])
    test_net = GAT(input_size=2, output_size=1, K=1, adj=test_adj)

    # Explicitly set the parameters for testing
    test_net.W = torch.nn.Parameter(torch.FloatTensor([[[2, 2], [3, 2]]]))
    test_net.a = torch.nn.Parameter(torch.FloatTensor([[[3, 2, 1, 1]]]))
    embedding = test_net.compute_embedding(test_nodes, i=1, k=0)

    # compare to expected value. Results were calculated by hand
    expected_attention = F.softmax(torch.FloatTensor([19, 32]), dim=0).view(2, 1)
    expected_output = sum(torch.FloatTensor([[4, 5],[10, 12]]) * expected_attention)

    # Add some tolerance when comparing
    try:
        assert torch.allclose(embedding, expected_output)
        print("SUCCESS: embedding matches expected values")
    except Exception as e:
        print("FAILED: ", e)

test_variables_change(net, nodes, outputs)
test_compute_embedding()

train(net, nodes, outputs, lr=0.001, iters=1000)
print(net.forward(nodes))
