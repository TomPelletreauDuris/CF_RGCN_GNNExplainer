import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Entities, Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj
import time
from sklearn import model_selection
import sys
sys.path.append('../')

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

from model.GCN import GCN, normalize_adjacency_matrix, RGCN, RGCNModel,RGCNModel_sparse
from model.Data_loading import load_cora_dataset, load_tree_dataset
from model.Preprocessing import create_feature_matrix, create_adjacency_matrix, create_target_vector, partition_dataset

# Functions to compute loss and accuracy
def compute_loss(y_pred, y_true):
    return F.nll_loss(y_pred, y_true)

def compute_accuracy(y_pred, y_true):
    return (y_pred == y_true).sum().item() / y_true.size(0)

def get_the_model():
    """
    Returns the trained model.

    Returns:
    model (GCN): The trained GCN model.
    num_nodes (int): The number of nodes in the graph.
    num_features (int): The number of features per node in the graph.
    num_labels (int): The number of labels in the graph.
    X (torch.sparse.FloatTensor): The feature matrix of the graph.
    A (torch.sparse.FloatTensor): The adjacency matrix of the graph.
    A_hat (torch.sparse.FloatTensor): The normalized adjacency matrix of the graph.
    y_true (torch.Tensor): The true labels of the nodes in the graph.
    train_idx (torch.Tensor): The indices of the training nodes.
    test_idx (torch.Tensor): The indices of the test nodes.
    W1 (torch.Tensor): The weight matrix of the first Graph Convolution Layer.
    W2 (torch.Tensor): The weight matrix of the second Graph Convolution Layer.
    """
    # Load and preprocess the data
    nodes, features, labels, edges_reindexed = load_cora_dataset('../data/')
    num_nodes = len(nodes)
    num_features = features.shape[1]
    num_labels = len(np.unique(labels))

    X = create_feature_matrix(features, num_nodes, num_features)
    A = create_adjacency_matrix(edges_reindexed, num_nodes)
    y_true, _ = create_target_vector(labels, num_nodes)
    train_idx, test_idx = partition_dataset(num_nodes)

    # Normalize the adjacency matrix (using the normalization function from Model_GCN.py)
    A_hat = normalize_adjacency_matrix(A.to_dense())
    A_hat = A_hat.to_sparse()

    # Initialize the GCN model
    model = GCN(num_features, 32, num_labels)
   
    # Hyperparameters
    learning_rate = 0.01
    num_epoch = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train = torch.zeros_like(X.to_dense())
    X_train[train_idx] = X.to_dense()[train_idx]

    # Training loop
    for epoch in range(1, num_epoch+1):
        model.train()
        
        y_pred = model(X_train, A_hat)
        loss = compute_loss(y_pred[train_idx], y_true[train_idx])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = compute_accuracy(y_pred.argmax(dim=1)[train_idx], y_true[train_idx])
        loss = float(loss) # release memory of computation graph
        print(f'Epoch {epoch:3d} - Loss: {loss:0.4f}, Acc: {acc:0.4f}')

    # Testing
    model.eval()
    with torch.no_grad():
        y_pred = model(X, A_hat)
        test_loss = compute_loss(y_pred[test_idx], y_true[test_idx])
        test_acc = compute_accuracy(y_pred.argmax(dim=1)[test_idx], y_true[test_idx])
        loss = float(loss) # release memory of computation graph
        print(f'Test Loss: {test_loss.item():0.4f}, Test Acc: {test_acc:0.4f}')

    W1 = model.conv1.W.data  # Assuming conv1 is your first Graph Convolution Layer
    W2 = model.conv2.W.data  # Assuming conv2 is your second Graph Convolution Layer


    return model, num_nodes, num_features, num_labels, X, A, A_hat, y_true, y_pred, train_idx, test_idx, W1, W2   


def main():

    """ # Load and preprocess the data
    adj, features, labels, idx_train, idx_test = load_tree_dataset('../data/')
    
    
    num_nodes = len(labels)
    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    
    print(adj.shape)
    print(adj.to_dense().shape)
    # print(y_true.shape)
    # print(y_true[idx_train].shape)
    """


    """ # Load and preprocess the data
    nodes, features, labels, edges_reindexed = load_cora_dataset('../data/')
    num_nodes = len(nodes)
    num_features = features.shape[1]
    num_labels = len(np.unique(labels))

    X = create_feature_matrix(features, num_nodes, num_features)
    A = create_adjacency_matrix(edges_reindexed, num_nodes)
    y_true, _ = create_target_vector(labels, num_nodes)
    train_idx, test_idx = partition_dataset(num_nodes)

    # Normalize the adjacency matrix (using the normalization function from Model_GCN.py)
    A_hat = normalize_adjacency_matrix(A.to_dense())
    A_hat = A_hat.to_sparse()
    
    print(A_hat.shape)
    print(A_hat.to_dense().shape)
    print(y_true.shape)
    print(y_true[train_idx].shape)

    # Initialize the GCN model
    model = GCN(num_features, 32, num_labels)
    print(model.conv1.W.shape)

    # Hyperparameters
    learning_rate = 0.01
    num_epoch = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train = torch.zeros_like(X.to_dense())
    X_train[train_idx] = X.to_dense()[train_idx]

    print(f'X_train dense shape: {X_train.shape}')
    print(f'X_train shape: {X_train.shape}')
    print(f'Number of features: {num_features}')
    print(f'Model conv1.W shape: {model.conv1.W.shape}')
    print(f'A_hat shape: {A_hat.shape}')

    # Training loop
    for epoch in range(1, num_epoch+1):
        model.train()
        
        y_pred = model(X_train, A_hat)
        loss = compute_loss(y_pred[train_idx], y_true[train_idx])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = compute_accuracy(y_pred.argmax(dim=1)[train_idx], y_true[train_idx])
        loss = float(loss) # release memory of computation graph
        print(f'Epoch {epoch:3d} - Loss: {loss:0.4f}, Acc: {acc:0.4f}')

    # Testing
    model.eval()
    with torch.no_grad():
        y_pred = model(X, A_hat)
        test_loss = compute_loss(y_pred[test_idx], y_true[test_idx])
        test_acc = compute_accuracy(y_pred.argmax(dim=1)[test_idx], y_true[test_idx])
        loss = float(loss) # release memory of computation graph
        print(f'Test Loss: {test_loss.item():0.4f}, Test Acc: {test_acc:0.4f}')

    W = model.conv1.W.data  # Assuming conv1 is your first Graph Convolution Layer
    #save model
    torch.save(model.state_dict(), '../data/model.pt')
    # Save the weight matrix W and feature matrix X
    torch.save(W, '../data/model_2layer/weight_matrix.pt')
    torch.save(X, '../data/model_2layer/feature_matrix.pt')
    # Save the predictions for future use
    torch.save(y_pred, '../data/model_2layer/model_predictions.pt')
    #save the true labels for future use
    torch.save(y_true, '../data/model_2layer/true_labels.pt')
    #save the full_adjacency_matrix for future use
    torch.save(A, '../data/model_2layer/full_adjacency_matrix.pt')
    #save the train_idx for future use
    torch.save(train_idx, '../data/model_2layer/train_idx.pt')
    #save the test_idx for future use
    torch.save(test_idx, '../data/model_2layer/test_idx.pt') """
""" 
    # Specify the root directory where the dataset will be stored
    root_dir = 'data/AIFB'

    # Load the AIFB dataset
    transform = T.Compose([T.ToUndirected(), T.NormalizeFeatures()])
    dataset = Entities(root=root_dir, name='AIFB') # ,transform=transform)
    
    # cora = Planetoid(root='data/Planetoid', name='Cora', transform=transform)
    data = dataset[0]  # AIFB dataset consists of a single graph
    # print(dir(dataset))
    print(data.edge_index)
    print(data.edge_type)
    print(data.train_y)
    print(data.train_idx)
    #features
    print(data.x)
    num_classes = dataset.num_classes
    num_relations = dataset.num_relations
    num_nodes = data.num_nodes

    ## splitting dataset into train(80%) and test(20%)
    train_idx = data.train_idx
    test_idx = data.test_idx
    # val_idx = data.val_idx


    # Get the labels
    train_labels = data.train_y
    test_labels = data.test_y
    # category = dataset.predict_category
    # Accessing all labels
    # all_labels = data.nodes[category].data['label'] 
    print

    # Get the features
    # train_x = data.train_x
    # test_x = data.test_x

    # Get the edge_index and edge_type from the dataset
    edge_index = data.edge_index
    edge_type = data.edge_type

    A = to_dense_adj(edge_index).squeeze(0)
    # A = A.to_sparse()

    print(A[0:15, 0:15])
    #check how much ones are in the adjacency matrix
    print(torch.sum(A))

    #check if the matrix is symetric
    print((A.t() == A))

    # exit()

    #train and test edge_index
    train_edge_index = data.edge_index[:, train_idx]
    test_edge_index = data.edge_index[:, test_idx]

    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Number of classes: {num_classes}')
    print(f'Number of relations: {num_relations}')
    print(f'Number of training nodes: {len(train_idx)}')
    print(f'Number of test nodes: {len(test_idx)}')
    # print(f'Number of evaluation nodes: {len(val_idx)}')
    print(f"edge index: {edge_index.shape}")
    print(f"edge type: {edge_type.shape}")
    print(f"train edge index: {train_edge_index.shape}")
    print(f"test edge index: {test_edge_index.shape}")


    # configurations
    n_hidden = 16 # number of hidden units
    n_bases = 0 # use number of relations as number of bases
    n_hidden_layers = 0 # use 1 input layer, 1 output layer, no hidden layer
    n_epochs = 25 # epochs to train
    lr = 0.01 # learning rate
    l2norm = 0 # L2 norm coefficient

    # model_1 = RGCN2(num_nodes=num_features, h_dim=n_hidden, out_dim=num_classes, num_rels=num_relations, num_bases=n_bases, num_hidden_layers=n_hidden_layers )
    # Define the model
    # model = RGCN(num_entities=num_nodes, num_relations = num_relations, num_classes = num_classes)
    # model = RGCNModel(num_node_features = num_nodes, num_classes = num_classes, num_relations = num_relations, hidden_channels = n_hidden)
    model = RGCNModel_sparse(in_channels = num_nodes, hidden_channels = n_hidden, out_channels = num_classes, num_relations = num_relations)
                             
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

    # create embeddings (done in the RGCN class already)
    embed_layer = torch.nn.Embedding(num_nodes, n_hidden)
    input_entities = torch.arange(num_nodes)
    embeds = embed_layer(input_entities)
    print(embeds.shape)

    edge_weight = torch.ones(edge_index.shape[1])

    print("start training...")
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        # logits = model.forward(edge_index, edge_type)
        
        logits = model.forward(embeds, A.to_sparse(), edge_type)
        
        # logits = model.forward(embeds, edge_index, edge_type, edge_weight)
        print(logits.shape)
        loss = F.cross_entropy(logits[train_idx], train_labels)
        loss.backward()

        optimizer.step()

        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == train_labels)
        train_acc = train_acc.item() / len(train_idx)
        test_loss = F.cross_entropy(logits[test_idx], test_labels)
        test_acc = torch.sum(logits[test_idx].argmax(dim=1) == test_labels)
        test_acc = test_acc / len(test_idx)
        print("Epoch {:05d} | ".format(epoch) +
            "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
                train_acc, loss.item()) +
            "Test Accuracy: {:.4f} | Test loss: {:.4f}".format(
                test_acc, test_loss.item()))
        
    model.eval()
    time.sleep(1)

    for name, param in model.conv1.named_parameters():
        if name == 'weight':
            print(param.data)
            print(param.data.shape)

    W = model.conv1.weight.data  # Assuming conv1 is your first Graph Convolution Layer
    print(W.shape)
    W2 = model.conv2.weight.data  # Assuming conv2 is your second Graph Convolution Layer
    print(W2.shape)

    X = embeds
    print(X.shape)
    A = to_dense_adj(edge_index).squeeze(0)
    print(A.shape)
    A_hat = normalize_adjacency_matrix(A)
    A_hat = A_hat.to_sparse()
    y_true_train = train_labels
    y_true_test = test_labels
    y_pred = logits
    print(y_pred.shape)
    print(logits.shape)
    train_idx = train_idx
    test_idx = test_idx


    #save model

    
    torch.save(model.state_dict(), '../data/R_model_2layer_wweight/R-model.pt')
    # Save the weight matrix W and feature matrix X
    torch.save(W, '../data/R_model_2layer_wweight/weight_matrix.pt')
    torch.save(W2, '../data/R_model_2layer_wweight/weight_matrix2.pt')
    torch.save(X, '../data/R_model_2layer_wweight/feature_matrix.pt')
    torch.save(num_nodes, '../data/R_model_2layer_wweight/num_nodes.pt')
    torch.save(num_classes, '../data/R_model_2layer_wweight/num_classes.pt')
    torch.save(num_relations, '../data/R_model_2layer_wweight/num_relations.pt')
    # Save the predictions for future use
    torch.save(y_pred, '../data/R_model_2layer_wweight/model_predictions.pt')
    #save the true labels for future use
    torch.save(y_true_train, '../data/R_model_2layer_wweight/true_labels_train.pt')
    torch.save(y_true_test, '../data/R_model_2layer_wweight/true_labels_test.pt')
    #save the full_adjacency_matrix for future use
    torch.save(A, '../data/R_model_2layer_wweight/R-full_adjacency_matrix.pt')
    torch.save(edge_index, '../data/R_model_2layer_wweight/R-edge_index.pt')
    torch.save(edge_type, '../data/R_model_2layer_wweight/R-edge_type.pt')
    #save the train_idx for future use
    torch.save(train_idx, '../data/R_model_2layer_wweight/train_idx.pt')
    #save the test_idx for future use
    torch.save(test_idx, '../data/R_model_2layer_wweight/test_idx.pt')  """
    
"""
    torch.save(model.state_dict(), '../data/R-model_2layer/R-model.pt')
    # Save the weight matrix W and feature matrix X
    torch.save(W, '../data/R-model_2layer/weight_matrix.pt')
    torch.save(W2, '../data/R-model_2layer/weight_matrix2.pt')
    torch.save(X, '../data/R-model_2layer/feature_matrix.pt')
    torch.save(num_nodes, '../data/R-model_2layer/num_nodes.pt')
    torch.save(num_classes, '../data/R-model_2layer/num_classes.pt')
    torch.save(num_relations, '../data/R-model_2layer/num_relations.pt')
    # Save the predictions for future use
    torch.save(y_pred, '../data/R-model_2layer/model_predictions.pt')
    #save the true labels for future use
    torch.save(y_true_train, '../data/R-model_2layer/true_labels_train.pt')
    torch.save(y_true_test, '../data/R-model_2layer/true_labels_test.pt')
    torch.save(all_labels, '../data/R-model_2layer/true_labels.pt')
    #save the full_adjacency_matrix for future use
    torch.save(A, '../data/R-model_2layer/R-full_adjacency_matrix.pt')
    torch.save(edge_index, '../data/R-model_2layer/R-edge_index.pt')
    torch.save(edge_type, '../data/R-model_2layer/R-edge_type.pt')
    #save the train_idx for future use
    torch.save(train_idx, '../data/R-model_2layer/train_idx.pt')
    #save the test_idx for future use
    torch.save(test_idx, '../data/R-model_2layer/test_idx.pt')  """

if __name__ == "__main__":
    main()
