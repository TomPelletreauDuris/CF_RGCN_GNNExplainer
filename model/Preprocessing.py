import torch
import numpy as np
from model.Data_loading import load_cora_dataset


def create_feature_matrix(features, num_nodes, num_features):
    """
    Generates the node feature matrix.

    Parameters:
    features (array): The features array from the dataset.
    num_nodes (int): Number of nodes in the dataset.
    num_features (int): Number of features for each node.

    Returns:
    torch.Tensor: The feature matrix as a sparse float tensor.
    """
    X = torch.from_numpy(features).float()
    X = X.to_sparse()
    assert X.size()[0] == num_nodes
    assert X.size()[1] == num_features
    return X

def create_adjacency_matrix(edges_reindexed, num_nodes):
    """
    Generates the adjacency matrix for the graph.

    Parameters:
    edges_reindexed (array): Reindexed edges of the graph.
    num_nodes (int): Number of nodes in the graph.

    Returns:
    torch.Tensor: The adjacency matrix as a sparse float tensor.
    """
    rows = torch.arange(0, num_nodes, dtype=torch.long)
    cols = torch.arange(0, num_nodes, dtype=torch.long)
    for each in edges_reindexed:
        rows = torch.cat((rows, torch.tensor([each[0]], dtype=torch.long)))
        cols = torch.cat((cols, torch.tensor([each[1]], dtype=torch.long)))
    indices = torch.stack((rows, cols))
    values = torch.ones(indices.shape[1], dtype=torch.float32)
    A = torch.sparse.FloatTensor(indices, values, torch.Size([num_nodes, num_nodes]))
    return A

def create_target_vector(labels, num_nodes):
    """
    Generates the target vector with integer-encoded class labels.

    Parameters:
    labels (array): Labels from the dataset.
    num_nodes (int): Number of nodes in the dataset.

    Returns:
    torch.Tensor: The target vector.
    """
    y = {l: n for n, l in enumerate(np.unique(labels))}
    y_true = torch.zeros(num_nodes, dtype=torch.long)
    for i, each in enumerate(labels):
        y_true[i] += torch.tensor(y[each], dtype=torch.long)
    num_labels = len(np.unique(labels))
    return y_true, num_labels

def partition_dataset(num_nodes):
    """
    Partitions the dataset into training and testing sets.

    Parameters:
    num_nodes (int): Number of nodes in the dataset.

    Returns:
    tuple: Indices for training and testing sets.
    """
    num_train = round(0.8 * num_nodes)
    num_test = round(0.2 * num_nodes)
    mask = torch.randperm(num_nodes)
    train_idx = mask[:num_train]
    test_idx = mask[num_train:]
    return train_idx, test_idx

def main():
    """
    Main function to load and inspect the Cora dataset. Act as a test function.
    """
    torch.manual_seed(42)  # for reproducibility
    path = '../data/'

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

    # Generate the feature matrix, adjacency matrix, and target vector
    X = create_feature_matrix(features, num_nodes, num_features)
    A = create_adjacency_matrix(edges_reindexed, num_nodes)
    y_true, num_labels = create_target_vector(labels, num_nodes)
    train_idx, test_idx = partition_dataset(num_nodes)

    # Testing outputs
    print(f'Feature matrix (sparse): {X}')
    print(f'Adjacency matrix (sparse): {A}')
    print(f'Target vector: {y_true[:10]}')
    print(f'Number of unique labels: {num_labels}')
    print(f'Train indices: {train_idx[:5]}')
    print(f'Test indices: {test_idx[:5]}')

    #test if the adjacency matrix is symmetric
    print(A.to_dense() == A.to_dense().t())
    print(A.to_dense().sum())
    print(A.to_dense().t().sum())
    print(A.to_dense().sum() == A.to_dense().t().sum())

# Optional: Main execution block for testing
if __name__ == "__main__":
    main()
    

