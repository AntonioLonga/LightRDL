import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, SAGEConv
from sklearn.metrics import roc_auc_score
import os
import numpy as np


class LightRDL(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers,dropout_prob=0.3,DIM_EMB=1,pk=None):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout_prob = dropout_prob
        self.dim_emb = DIM_EMB -10
        self.pk = pk
        # Define the MLP for the first 1500 features of "drivers" nodes
        self.drivers_mlp = torch.nn.Sequential(
            Linear(self.dim_emb, 20),  # First layer: from 1500 features to 64
            torch.nn.ReLU(),
            Linear(20, 10)     # Second layer: from 64 to 10
        )

        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

        # Linear layers after convolution
        self.lin0 = Linear(hidden_channels, int(hidden_channels / 2))
        self.lin = Linear(int(hidden_channels / 2), out_channels)
        
        assert not pk == None

    def forward(self, x_dict, edge_index_dict):
        drivers_features = x_dict[self.pk]  
        first_1500_features = drivers_features[:, :self.dim_emb]  
        remaining_10_features = drivers_features[:, self.dim_emb:]

        mlp_output = self.drivers_mlp(first_1500_features)
        drivers_processed = torch.cat([mlp_output, remaining_10_features], dim=-1)
        x_dict[self.pk] = drivers_processed

        for conv in self.convs:
            residual_remaining_10 = x_dict[self.pk][:, -10:].clone()
            x_dict = conv(x_dict, edge_index_dict)
            conv_output_remaining_10 = x_dict[self.pk][:, -10:]  
            conv_output_remaining_10 = conv_output_remaining_10 + residual_remaining_10  
            
            x_dict[self.pk] = torch.cat([x_dict[self.pk][:, :-10], conv_output_remaining_10], dim=-1)

            # Apply non-linearity and dropout
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout_prob, training=self.training) for key, x in x_dict.items()}

        # Apply the final linear layers for "drivers"
        out = self.lin0(x_dict[self.pk])
        out = F.relu(out)
        out = self.lin(out)

        return out

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers,dropout_prob=0.3,DIM_EMB=None,pk=None):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.dropout_prob = dropout_prob
        self.pk = pk
        
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)
        assert not pk == None

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}            
            x_dict = {key: F.dropout(x, p=self.dropout_prob, training=self.training) for key, x in x_dict.items()}

        return self.lin(x_dict[self.pk])
    


def train(model,train_loader,device,optimizer,pk,criterion):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = criterion(out, batch.y_dict[pk].reshape(-1,1).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model,loader,device,pk,criterion):
    model.eval()
    with torch.no_grad():
        tmp_pred = []
        tmp_y = []
        for batch in loader:   
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)            
            loss = criterion(out, batch.y_dict[pk].reshape(-1,1).float())
            pred = F.sigmoid(out)
            tmp_y.extend(batch.y_dict[pk].detach().cpu().numpy().tolist())
            tmp_pred.extend(pred.detach().cpu().numpy().reshape(-1).tolist())
    return loss, tmp_y, tmp_pred
    #return loss,roc_auc_score(tmp_y,tmp_pred)


