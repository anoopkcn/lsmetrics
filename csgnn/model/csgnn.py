import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class CSGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_layers, dropout_rate=0.5):
        super(CSGNN, self).__init__()
        self.num_layers = num_layers

        # Initial node embedding
        self.node_embedding = nn.Linear(num_node_features, hidden_channels)

        # Graph Convolutional layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Edge embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(num_edge_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Output layers
        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, 1)  # For property prediction

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data, mode='regression'):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Initial node and edge embeddings
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        # Graph convolutions
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.dropout(x)

        # Global pooling
        graph_embedding = global_mean_pool(x, batch)

        if mode == 'encoder':
            return graph_embedding
        elif mode == 'regression':
            # Final layers for property prediction
            x = self.linear1(graph_embedding)
            x = F.relu(x)
            x = self.dropout(x)
            property_prediction = self.linear2(x)
            return property_prediction
        else:
            raise ValueError("Invalid mode. Use 'encoder' or 'regression'.")

    def encode(self, data):
        return self.forward(data, mode='encoder')

    def predict_property(self, data):
        return self.forward(data, mode='regression')
