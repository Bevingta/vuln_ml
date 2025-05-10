import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, RGCNConv, global_mean_pool, GatedGraphConv
from sklearn.model_selection import train_test_split
import torch


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.5, model="gcn"):
        '''
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (2 for binary classification)
            dropout: Dropout probability
        '''
        super(GNNModel, self).__init__()
        self.dropout = dropout
        self.model = model.lower()
        if self.model == "gcn":
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif self.model == "rgcn":
            self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations=6)
            self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=6)
            self.conv3 = RGCNConv(hidden_dim, hidden_dim, num_relations=6)
        
        self.mlp = nn.Sequential(
                nn.Linear(hidden_dim + 5, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )


    def forward(self, data: Data):
        if self.model == "gcn":
            x, edge_index, batch = data.x, data.edge_index, data.batch

            # First layer
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Second layer
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            
            # Global pooling
            #x = global_mean_pool(x, batch)
            x = global_mean_pool(x, batch)
        
        elif self.model == "rgcn": 
            x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch

            x = self.conv1(x, edge_index, edge_type)
            x = F.relu(x)

            x = self.conv2(x, edge_index, edge_type)
            x = F.relu(x)

            x = self.conv3(x, edge_index, edge_type)
            x = F.relu(x)
            
            x = F.dropout(x, p=self.dropout, training=self.training)

            #x = global_mean_pool(x, batch)
            x = global_mean_pool(x, batch)

             # Retrieve and stack graph-level flags
            flags = torch.stack([g.graph_flags for g in data.to_data_list()]).to(x.device)
            x = torch.cat([x, flags], dim=1)  # Concatenate pooled vector + flags
        
        return self.mlp(x)  # Shape: [batch_size, 2]