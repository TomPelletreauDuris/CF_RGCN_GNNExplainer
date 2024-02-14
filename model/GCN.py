import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import math
from torch_geometric.nn import GCNConv, GATConv, RGCNConv
from functools import partial
import sys
sys.path.append('../')
import torch_geometric
print(torch_geometric.__version__)

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Parameter as Param

import torch_geometric.backend
import torch_geometric.typing
from torch_geometric import is_compiling
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    SparseTensor,
    pyg_lib,
    torch_sparse,
)
from torch_geometric.utils import index_sort, one_hot, scatter, spmm
from torch_geometric.utils.sparse import index2ptr


# Import preprocessing functions from Preprocessing.py
from model.Data_loading import load_cora_dataset
from model.Preprocessing import create_feature_matrix, create_adjacency_matrix, create_target_vector
from model.utils import normalize_adjacency_matrix


import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_

class RGCNConv_spare(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, num_bases=None, num_blocks=None, bias=True):
        super(RGCNConv_spare, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks

        if num_bases is not None:
            self.weight = Parameter(torch.Tensor(num_bases, in_channels, out_channels))
            self.comp = Parameter(torch.Tensor(num_relations, num_bases))
        elif num_blocks is not None:
            assert in_channels % num_blocks == 0 and out_channels % num_blocks == 0
            self.weight = Parameter(torch.Tensor(num_relations, num_blocks, in_channels // num_blocks, out_channels // num_blocks))
        else:
            self.weight = Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.num_bases is not None or self.num_blocks is None:
            xavier_uniform_(self.weight)
            if self.num_bases is not None:
                xavier_uniform_(self.comp)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)
    
    def forward(self, x, A, edge_type):
        out = torch.zeros((x.size(0), self.out_channels), device=x.device)
        for rel in range(self.num_relations):
            rel_mask = edge_type == rel
            if self.num_bases is not None:
                W = torch.matmul(self.comp[rel], self.weight.view(self.num_bases, -1))
                W = W.view(self.in_channels, self.out_channels)
            elif self.num_blocks is not None:
                W = self.weight[rel].view(-1, self.out_channels)
            else:
                W = self.weight[rel]

            rel_adj = A * rel_mask.float()
            h = torch.matmul(rel_adj, x)
            h = torch.matmul(h, W)
            out += h
        
        if self.bias is not None:
            out += self.bias
        return out


class RGCNModel_sparse(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_bases=None, num_blocks=None):
        super(RGCNModel_sparse, self).__init__()
        # First RGCN layer
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases, num_blocks=num_blocks)
        # Second RGCN layer
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations, num_bases=num_bases, num_blocks=num_blocks)
        # Optional: Add a non-linearity between RGCN layers
        self.relu = nn.ReLU()

    def forward(self, x, A, edge_type):
        # Apply first RGCN layer
        x = self.conv1(x, A, edge_type)
        x = self.relu(x)  # Apply non-linearity
        # Apply second RGCN layer
        x = self.conv2(x, A, edge_type)
        return x




#Extended RGCNConv to be able to use edge_weight
class ExtendedRGCNConv(RGCNConv):
    """
    Extended RGCNConv to be able to use edge_weight
    Inspired from Matthias Fey Creator of PyG (PyTorch Geometric) in forum
    
    """
    def __init__(self, in_channels, out_channels, num_relations):
        super(ExtendedRGCNConv, self).__init__(in_channels, out_channels, num_relations)

    def forward(self, x, edge_index, edge_weight, edge_type):
        x_l= None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r=x_l
        if isinstance(x, tuple):
            x_r = x[1]
        # Initialize the output tensor
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        for i in range(self.num_relations):
            mask = edge_type == i
            # Ensure edge_weight[mask] is passed as part of the message kwargs
            out += self.propagate(edge_index[:, mask], x=x, size=None, edge_type=edge_type[mask], edge_weight=edge_weight[mask])
        return out
    
    def message(self, x_j, edge_index_i, edge_weight=None, edge_type=None, size_i=None):
        # Apply edge weights if provided
        if edge_weight is not None:
            weighted_message = edge_weight.view(-1, 1) * x_j
        else:
            weighted_message = x_j

        # Use edge_type to select the appropriate weights
        # Ensure edge_type is used correctly, possibly as an index
        if edge_type is not None:
            # Assuming edge_type is used to index a weight matrix specific to the edge type
            return weighted_message @ self.weight[edge_type]
        else:
            return weighted_message



class RGCNModel(nn.Module):
    def __init__(self, num_node_features, num_classes, num_relations, hidden_channels):
        super(RGCNModel, self).__init__()

        self.conv1 = ExtendedRGCNConv(in_channels=num_node_features, out_channels=hidden_channels, num_relations=num_relations)
        self.conv2 = ExtendedRGCNConv(in_channels=hidden_channels, out_channels=num_classes, num_relations=num_relations)

    def forward(self, x, edge_index, edge_type, edge_attr):
        x = self.conv1(x=x, edge_index=edge_index, edge_type=edge_type,edge_weight=edge_attr)
        x = F.relu(x)
        x = self.conv2(x=x, edge_index=edge_index, edge_type=edge_type,edge_weight=edge_attr)
        return F.log_softmax(x, dim=1)
    
# Define the Graph Convolution Layer
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(in_features, out_features), requires_grad=True)
        nn.init.xavier_uniform_(self.W)

    def forward(self, X: torch.sparse.Tensor, A: torch.sparse.Tensor) -> torch.Tensor:
        # print(f"Shape of X: {X.shape}")  
        # print(f"Shape of self.W: {self.W.shape}")  
        support = torch.sparse.mm(X, self.W)
        return torch.sparse.mm(A, support)

# Define the GCN Model
class GCN(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.conv1 = GraphConvolutionLayer(in_features, hidden_size)
        self.conv2 = GraphConvolutionLayer(hidden_size, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, X: torch.sparse.Tensor, A: torch.sparse.Tensor) -> torch.Tensor:
        x = self.conv1(X, A)
        # print(f"First conv layer in model : {x}")
        x = self.relu(x)
        # print(f"First relu layer in model : {x}")
        x = self.dropout(x)
        print(f"First dropout layer in model : {x}")
        x = self.conv2(x, A)
        x = self.softmax(x)
        return x

class GCN3L(nn.Module):
    """
    3-layer GCN used in GNN Explainer synthetic tasks
    """
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GCN3L, self).__init__()
        self.num_layers = 3
        self.gc1 = GraphConvolutionLayer(nfeat, nhid)
        self.gc2 = GraphConvolutionLayer(nhid, nhid)
        self.gc3 = GraphConvolutionLayer(nhid, nout)
        self.lin = nn.Linear(nhid + nhid + nout, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, adj)
        x = self.lin(torch.cat((x1, x2 ,x3), dim=1))
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    

#Same as the original repo to replicate the results
class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        #print(adj.shape)
        #print(input.shape)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

#Same as the original repo to replicate the results
class GCN3Layer_PyG(nn.Module):
    """
    3-layer GCN
    """
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GCN3Layer_PyG, self).__init__()
        self.num_layers = 3
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.gc3 = GCNConv(nhid, nhid)
        self.lin = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        print(x.size())
        print(edge_index.size())
        x1 = F.relu(self.gc1(x, edge_index))
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, edge_index))
        #x2 = F.dropout(x2, self.dropout, training=self.training)
        x3= self.gc3(x2, edge_index)
        x = self.lin(x3) #x = self.lin(torch.cat((x1, x2 ,x3), dim=1))
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    
#Same as the original repo to replicate the results   
class GCNSynthetic(nn.Module):
    """
    3-layer GCN used in GNN Explainer synthetic tasks
    """
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GCNSynthetic, self).__init__()
        self.num_layers = 3
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nout)
        # self.lin = nn.Linear(nhid, nclass)
        self.lin = nn.Linear(nhid + nhid + nout, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, adj)
        x = self.lin(torch.cat((x1, x2 ,x3), dim=1))
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

# NOT used as we use directly the Pytorch Geometric one
""" class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['rel_type'] * self.in_feat + edges.src['id']
                return {'msg': embed[index] * edges.data['norm']}
        else:
            def message_func(edges):
                w = weight[edges.data['rel_type']]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func) """

#Our simple RGCN implementation
class RGCN(nn.Module):
    """ 
    Simple RGCN model with 2 layers. 

    Args:
    num_entities (int): Number of entities in the graph.
    num_relations (int): Number of relations in the graph.
    num_classes (int): Number of classes in the graph.
    """
    def __init__(self, num_entities, num_relations, num_classes):
        super(RGCN, self).__init__()
        self.embed = torch.nn.Embedding(num_entities, 16)
        self.conv1 = RGCNConv(16, 16, num_relations)
        self.conv2 = RGCNConv(16, num_classes, num_relations)

    def forward(self, edge_index, edge_type):
        x = self.embed.weight
        x = self.conv1(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return torch.log_softmax(x, dim=1)
    
class RGCN_withmask(nn.Module):
    """ 
    Simple RGCN model with 2 layers. 

    Args:
    num_entities (int): Number of entities in the graph.
    num_relations (int): Number of relations in the graph.
    num_classes (int): Number of classes in the graph.
    """
    def __init__(self, num_entities, num_relations, num_classes, mask):
        super(RGCN_withmask, self).__init__()
        self.embed = torch.nn.Embedding(num_entities, 16)
        self.conv1 = RGCNConv(16, 16, num_relations)
        self.conv2 = RGCNConv(16, num_classes, num_relations)

    def forward(self, edge_index, edge_type):
        x = self.embed.weight
        x = self.conv1(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return torch.log_softmax(x, dim=1)
    
class RGCN_Adj(nn.Module):
    """
    Simple RGCN model with 2 layers. Which can take a sparse adjacency matrix as input.
    like A = to_dense_adj(edge_index).squeeze(0) as edge_index and edge_type used normally.
    
    """

#Not used but could be useful
class RGCN2(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels,
                 num_bases=-1, num_hidden_layers=1):
        super(RGCN2, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        features = torch.arange(self.num_nodes)
        return features

    def build_input_layer(self):
        return RGCN(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True)

    def build_hidden_layer(self):
        return RGCN(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCN(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                         activation=partial(F.softmax, dim=1))

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        for layer in self.layers:
            layer(g)
        return g.ndata.pop('h')

def main():
    """
    Main function to lunch our GCN model. Act as a test function.
    """

    # Load and preprocess the data
    nodes, features, labels, edges_reindexed = load_cora_dataset('../data/')
    num_nodes = len(nodes)
    num_features = features.shape[1]
    num_labels = len(np.unique(labels))

    X = create_feature_matrix(features, num_nodes, num_features)
    A = create_adjacency_matrix(edges_reindexed, num_nodes)
    y_true, _ = create_target_vector(labels, num_nodes) #could be useful later

    # Normalize the adjacency matrix
    A_hat = normalize_adjacency_matrix(A.to_dense())  # Make sure A is dense for normalization
    A_hat = A_hat.to_sparse()  # 

    # Initialize the GCN model
    model = GCN(num_features, 32, num_labels)

    # Sample forward pass
    y_pred = model(X, A_hat)

    # Print the output for inspection (optional)
    print("Output from GCN model:", y_pred)

if __name__ == "__main__":
    main()
