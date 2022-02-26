import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        nn.init.xavier_normal_(self.weight.data)
        nn.init.zeros_(self.bias.data)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=-1, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output
