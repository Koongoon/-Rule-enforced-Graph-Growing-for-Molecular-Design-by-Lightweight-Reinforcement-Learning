import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomGCN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32):
        super(CustomGCN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.layer1(x)
        x = self.ln1(x)
        x = F.relu(x, inplace=False)
        x = self.layer2(x)
        x = self.ln2(x)
        x = F.relu(x, inplace=False)

        row, col = edge_index
        aggr = torch.zeros_like(x)
        aggr = aggr.index_add(0, row, x[col])
        x = x + aggr
        return x
