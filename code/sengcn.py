import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # GCN layer: H = A * X * W
        out = torch.matmul(adj, x)  # Matrix multiplication of adj matrix and input features
        out = self.fc(out)          # Apply the linear transformation
        return F.relu(out)          # ReLU activation

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        # Define the layers
        self.gcn1 = GCNLayer(in_features, hidden_features)
        self.gcn2 = GCNLayer(hidden_features, out_features)

    def forward(self, x, adj):
        # Apply the first GCN layer
        x = self.gcn1(x, adj)
        # Apply the second GCN layer
        x = self.gcn2(x, adj)
        return x

class SenGCN(nn.Module):
    def __init__(self, bert_output_dim, gcn_hidden_dim, gcn_output_dim):
        super(SenGCN, self).__init__()
        self.gcn = GCN(bert_output_dim, gcn_hidden_dim, gcn_output_dim)

    def forward(self, bert_embeddings, adj_matrix):
        # Apply the GCN layers
        gcn_output = self.gcn(bert_embeddings, adj_matrix)
        return gcn_output