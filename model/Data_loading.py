import numpy as np
import torch
import pickle


def load_cora_dataset(path):
    """
    Loads the Cora dataset from the given path.

    Parameters:
    path (str): Path to the dataset directory.

    Returns:
    tuple: Returns nodes, features, labels, and reindexed edges of the dataset.
    """
    # Load data
    data = np.genfromtxt(path + "cora.content", dtype=str)
    edges = np.genfromtxt(path + "cora.cites", dtype=int)

    # Extract features, labels, and nodes
    features = data[:, 1:-1].astype(int)
    labels = data[:, -1]
    nodes = data[:, 0].astype(int)

    # Create a node-to-index mapping
    n2i = {n: i for i, n in enumerate(nodes)}
    
    # Reindex edges
    edges_reindexed = np.array([[n2i[source], n2i[target]] for source, target in edges])

    return nodes, features, labels, edges_reindexed

def load_tree_dataset(path):
    """
    Loads the Tree dataset from the given path.

    Parameters:
    path (str): Path to the dataset directory.

    Returns:
    adj (torch.Tensor): The adjacency matrix of the dataset.
    features (torch.Tensor): The features of the dataset.
    labels (torch.Tensor): The labels of the dataset.
    idx_train (torch.Tensor): The indices of the training nodes.
    idx_test (torch.Tensor): The indices of the test nodes.    
    """
        
    with open(path + "syn4.pickle", "rb") as f:
        data = pickle.load(f)

    adj = torch.Tensor(data["adj"]).squeeze()  # Does not include self loops
    features = torch.Tensor(data["feat"]).squeeze()
    labels = torch.tensor(data["labels"]).squeeze()
    idx_train = torch.tensor(data["train_idx"])
    idx_test = torch.tensor(data["test_idx"])
                            
    return adj, features, labels, idx_train, idx_test

def load_BA_dataset(path):
    """
    Loads the Tree dataset from the given path.

    Parameters:
    path (str): Path to the dataset directory.

    Returns:
    adj (torch.Tensor): The adjacency matrix of the dataset.
    features (torch.Tensor): The features of the dataset.
    labels (torch.Tensor): The labels of the dataset.
    idx_train (torch.Tensor): The indices of the training nodes.
    idx_test (torch.Tensor): The indices of the test nodes.    
    """
    print('load BA dataset')
    with open(path + "syn1.pickle", "rb") as f:
        data = pickle.load(f)

    adj = torch.Tensor(data["adj"]).squeeze()  # Does not include self loops
    features = torch.Tensor(data["feat"]).squeeze()
    labels = torch.tensor(data["labels"]).squeeze()
    idx_train = torch.tensor(data["train_idx"])
    idx_test = torch.tensor(data["test_idx"])
                            
    return adj, features, labels, idx_train, idx_test

def load_AIFB_dataset(path):
    """
    Loads the AIFB dataset from the given path.
    """
    
    A = torch.load('data/R-model_2layer/R-full_adjacency_matrix.pt')
    X = torch.load('data/R-model_2layer/feature_matrix.pt')
    true_labels_train = torch.load('data/R-model_2layer/true_labels_train.pt')
    true_labels_test = torch.load('data/R-model_2layer/true_labels_test.pt')
    logits = torch.load('data/R-model_2layer/model_predictions.pt')
    print(logits.shape)
    train_idx = torch.load('data/R-model_2layer/train_idx.pt')
    test_idx = torch.load('data/R-model_2layer/test_idx.pt')
    weight_matrix_1 = torch.load('data/R-model_2layer/weight_matrix.pt')
    weight_matrix_2 = torch.load('data/R-model_2layer/weight_matrix2.pt')
    num_nodes = torch.load('data/R-model_2layer/num_nodes.pt')
    num_relations = torch.load('data/R-model_2layer/num_relations.pt')
    num_classes = torch.load('data/R-model_2layer/num_classes.pt')
    edge_index = torch.load('data/R-model_2layer/R-edge_index.pt')
    edge_type = torch.load('data/R-model_2layer/R-edge_type.pt')

    return A, X, true_labels_train, true_labels_test, logits, train_idx, test_idx, weight_matrix_1, weight_matrix_2, num_nodes, num_relations, num_classes, edge_index, edge_type

def main():
    """
    Main function to load and inspect the Cora dataset. Act as a test function.
    """
    torch.manual_seed(42)  # for reproducibility
    path = './data/'

    # Load the Cora dataset
    nodes, features, labels, edges_reindexed = load_cora_dataset(path)

    # Inspect the data
    num_nodes = len(nodes)
    num_edges = len(edges_reindexed)
    num_features = len(features[0])

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Number of features: {num_features}\n")
    
    for i in range(5):
        print(f"Node ID: {nodes[i]}")
        print(f"Node features: {features[i]}")
        print(f"Node label: {labels[i]}\n")
        print(f"Edges: \n{edges_reindexed[:5]}")

if __name__ == "__main__":
    main()
