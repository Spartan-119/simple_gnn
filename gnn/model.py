import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv as gcn

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = gcn(in_channels, hidden_channels)
        self.conv2 = gcn(hidden_channels, out_channels)
        self.linear = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.linear(x)
        return F.log_softmax(x, dim = 1)