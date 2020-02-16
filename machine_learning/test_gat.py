import torch
from torchsummary import summary
from gat import train, GAT

#if __name__ == "__main__":
input_size = 2 # number of input features per node
output_size = 1 # number of output features per node
k = 1 # number of attention mechanisms

# adjacency matrix
edges = torch.LongTensor([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]])
ones = torch.ones(edges.size(0))
adj_matrix = torch.sparse.IntTensor(edges.t(), ones)
print("Is this even changing?")
# adj_matrix = adj_matrix.to_dense()

# input values
nodes = torch.FloatTensor([
    [1, 2],
    [2, 2],
    [-1, 1],
    [3, 3]
])
outputs = torch.FloatTensor([6, 1.5, 7.5, -1])

net = GAT(input_size, output_size, k, adj_matrix)
# train(net, nodes, outputs, lr=0.001, iters=1000)
# print(net.forward(nodes))
