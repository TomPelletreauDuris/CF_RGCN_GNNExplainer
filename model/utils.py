import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, to_dense_adj, subgraph

def normalize_adjacency_matrix(adj):
    A_tilde = adj + torch.eye(adj.shape[0])
    D_tilde = torch.diag(A_tilde.sum(1))
    D_tilde_exp = D_tilde.pow(-0.5)
    D_tilde_exp[torch.isinf(D_tilde_exp)] = 0
    norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
    return norm_adj


def get_neighbourhood_fromRE(node_idx, adjacency_matrix, n_hops, features, labels):
    edge_index, _ = dense_to_sparse(adjacency_matrix)
    edge_index = edge_index.long()
    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index) 
    edge_subset_relabel = subgraph(edge_subset[0], edge_index, relabel_nodes=True)       # Get relabelled subset of edges
    print("Edge subset: {}".format(edge_subset))
    # edge_subset = torch.tensor(edge_subset, dtype=torch.long)

    sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
    print(sub_adj)
    sub_feat = features.to_dense()[edge_subset[0], :]
    sub_labels = labels[edge_subset[0]]
    new_index = np.array([i for i in range(len(edge_subset[0]))])
    node_dict = dict(zip(edge_subset[0].numpy(), new_index))     
	
   
    print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
    print("Num edges in subgraph: {}".format(sub_adj.sum() / 2))
    print("Num features in subgraph: {}".format(sub_feat.shape[0]))
	
    return sub_adj, sub_feat, sub_labels, node_dict

def get_neighbourhood_fromRGCN(node_idx, edge_index, edge_type, n_hops, features):
    # Use k_hop_subgraph to extract the k-hop subgraph
    subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(node_idx, n_hops, edge_index, relabel_nodes=True)
    
    # Filter edge types for the edges included in the subgraph
    subgraph_edge_types = edge_type[edge_mask]
    
    # Optionally convert to dense adjacency matrix and extract corresponding features
    # This step is optional and depends on whether you need the dense adjacency matrix or not
    # sub_adj = to_dense_adj(edge_index_sub).squeeze()
    sub_feat = features[subset, :]
    
    # Create a node dictionary for mapping original node indices to new indices in the subgraph
    new_index = torch.arange(len(subset))
    node_dict = dict(zip(subset.numpy(), new_index.numpy()))

    print(f"Num nodes in subgraph: {len(subset)}")
    print(f"Num edges in subgraph: {edge_index_sub.size(1)}")
    print(f"Num features in subgraph: {sub_feat.shape[0]}")
    
    # Return the relabeled edge indices, filtered edge types, features of the subgraph nodes, and the node mapping dictionary
    return edge_index_sub, subgraph_edge_types, sub_feat, node_dict

def select_node_by_confidence(predictions, true_labels, adij, threshold=1):
    """
    Select a node that is well-predicted with a confidence above the threshold.

    Args:
    predictions (torch.Tensor): The model's predictions (logits or probabilities).
    true_labels (torch.Tensor): The true labels of the nodes.
    threshold (float): The confidence threshold.

    Returns:
    int: The index of a well-predicted node, or None if no such node is found.
    """
    # Apply softmax if predictions are logits
    print(predictions[0:5])
    if predictions.shape[1] > 1:  # Assuming predictions are in shape [num_nodes, num_classes]
        print(predictions.shape)
        probabilities = F.softmax(predictions, dim=1)
        print(probabilities.shape)
        print(probabilities[0:5])
    else:
        probabilities = predictions  # Assuming predictions are already probabilities

    confidences, predicted_classes = torch.max(probabilities, dim=1)
    print(confidences)
    print(predicted_classes)
    #print the true labels
    print(true_labels)
    correct_predictions = predicted_classes == true_labels
    print(correct_predictions)

    # Filter nodes that are correctly predicted with high confidence
    high_confidence_correct = (confidences > threshold) & correct_predictions
    print(f'high confidence :',high_confidence_correct)
    selected_indices = high_confidence_correct.nonzero(as_tuple=True)[0]
    if len(selected_indices) > 0:
        # Returning a randonly sampled well-predicted node
        print("Found nodes with the specified confidence criteria.")
        # selected_node = selected_indices[torch.randint(len(selected_indices), (1,))].item()
        # return selected_node, predicted_classes[selected_node].item(), confidences[selected_node].item()
        # Returning a sampled node with at least 2 neighboors
        for i in range(len(selected_indices)):
            selected_node = selected_indices[i].item()
            neighboors, _, _, _ = get_neighbourhood(selected_node, adij, 2, torch.ones(adij.shape[0]).unsqueeze(1), torch.ones(adij.shape[0]))
            if len(neighboors) >= 4:
                return selected_node, predicted_classes[selected_node].item(), confidences[selected_node].item()
        print("Found a node with the specified number of neighboors.")
        return selected_node, predicted_classes[selected_node].item(), confidences[selected_node].item()
    else:
        print("No nodes found with the specified confidence criteria.")
        return None
    
""" def select_node_by_confidence(predictions, threshold=0.3, highest=True):
    
    confidences, predicted_classes = torch.max(predictions, dim=1)
    print(confidences)
    print(predicted_classes)
    if highest:
        selected_indices = (confidences > threshold).nonzero(as_tuple=True)[0]
    else:
        selected_indices = (confidences < threshold).nonzero(as_tuple=True)[0]
    
    if len(selected_indices) == 0:
        # Handle the case when no nodes meet the threshold condition
        print("No nodes found with the specified confidence criteria.")
        return None, None, None

    if highest:
        selected_node = selected_indices[torch.argmax(confidences[selected_indices])].item()
    else:
        selected_node = selected_indices[torch.argmin(confidences[selected_indices])].item()

    return selected_node, predicted_classes[selected_node].item(), confidences[selected_node].item() """

def get_nodes_w_confidence(predictions, true_labels, adj, threshold=0.99, min_neighbors=3):
    """
    Select nodes that are well-predicted above or below a certain confidence threshold.

    Args:
        predictions (torch.Tensor): The model's predictions (logits or probabilities).
        true_labels (torch.Tensor): The true labels of the nodes.
        adj (torch.Tensor): The adjacency matrix of the graph.
        threshold (float): The confidence threshold.
        min_neighbors (int): Minimum number of neighbors a node must have.

    Returns:
        torch.Tensor: Indices of nodes above the threshold with enough neighbors.
        torch.Tensor: Indices of nodes below the threshold with enough neighbors.
    """
    if predictions.shape[1] > 1:
        probabilities = F.softmax(predictions, dim=1)
    else:
        probabilities = predictions

    confidences, predicted_classes = torch.max(probabilities, dim=1)
    correct_predictions = predicted_classes == true_labels

    high_confidence_correct = (confidences > threshold) & correct_predictions
    low_confidence = ~high_confidence_correct  # Nodes below the threshold or incorrectly predicted

    high_conf_indices = high_confidence_correct.nonzero(as_tuple=False).squeeze()
    low_conf_indices = low_confidence.nonzero(as_tuple=False).squeeze()

    # Filter for nodes with at least min_neighbors neighbors
    high_conf_indices_filtered = [idx for idx in high_conf_indices if len(get_2_hop_neighborhood(idx, adj)) >= min_neighbors]
    low_conf_indices_filtered = [idx for idx in low_conf_indices if len(get_2_hop_neighborhood(idx, adj)) >= min_neighbors]

    return torch.tensor(high_conf_indices_filtered), torch.tensor(low_conf_indices_filtered)

def select_node_by_prediction_error(predictions, true_labels):
    """
    Select a node with an unexpected or incorrect prediction.
    """
    _, predicted_classes = torch.max(predictions, dim=1)
    incorrect_predictions = (predicted_classes != true_labels).nonzero(as_tuple=True)[0]
    if len(incorrect_predictions) > 0:
        # If there are incorrect predictions, select one.
        selected_node = incorrect_predictions[0].item()  # Taking the first one for simplicity
        return selected_node, predicted_classes[selected_node].item(), true_labels[selected_node].item()
    else:
        # If all predictions are correct, return None
        return None, None, None
    
def calculate_entropy(predictions):
    """
    Calculate the entropy of the predictions for each node.
    """
    # Apply softmax to get probabilities
    probabilities = F.softmax(predictions, dim=1)
    # Calculate entropy
    log_probabilities = torch.log(probabilities + 1e-9)  # Add a small number to prevent log(0)
    entropy = -torch.sum(probabilities * log_probabilities, dim=1)
    return entropy

def select_node_by_uncertainty(predictions):
    """
    Select a node based on the highest uncertainty in its predictions.
    Uncertainty is measured as the entropy of the prediction distribution.
    """
    entropy = calculate_entropy(predictions)
    # Node with the maximum entropy is the most uncertain one
    selected_node = torch.argmax(entropy).item()
    return selected_node, entropy[selected_node].item()

def get_2_hop_neighborhood(node_id, adjacency_matrix):
    # Ensure the node_id is an integer, not a tensor
    if isinstance(node_id, torch.Tensor):
        node_id = node_id.item()

    # Convert the sparse adjacency matrix to dense if it's in sparse format
    if adjacency_matrix.is_sparse:
        adjacency_matrix = adjacency_matrix.to_dense()

    # Get 1-hop neighbors
    one_hop_neighbors = set(adjacency_matrix[node_id].nonzero().squeeze(1).tolist())

    # Get 2-hop neighbors
    two_hop_neighbors = set()
    for neighbor in one_hop_neighbors:
        neighbors_of_neighbor = adjacency_matrix[neighbor].nonzero().squeeze(1).tolist()
        two_hop_neighbors.update(neighbors_of_neighbor)

    # Combine 1-hop and 2-hop neighbors and remove the original node /!\
    neighborhood = one_hop_neighbors.union(two_hop_neighbors)
    #neighborhood.discard(node_id)

    neighborhood = list(neighborhood)

    return neighborhood


def get_4_hop_neighborhood(node_id, adjacency_matrix):
    # Ensure the node_id is an integer, not a tensor
    if isinstance(node_id, torch.Tensor):
        node_id = node_id.item()

    # Convert the sparse adjacency matrix to dense if it's in sparse format
    if adjacency_matrix.is_sparse:
        adjacency_matrix = adjacency_matrix.to_dense()

    # Get 1-hop neighbors
    one_hop_neighbors = set(adjacency_matrix[node_id].nonzero().squeeze(1).tolist())

    # Get 2-hop neighbors
    two_hop_neighbors = set()
    for neighbor in one_hop_neighbors:
        neighbors_of_neighbor = adjacency_matrix[neighbor].nonzero().squeeze(1).tolist()
        two_hop_neighbors.update(neighbors_of_neighbor)

    # Get 3-hop neighbors
    three_hop_neighbors = set()
    for neighbor in two_hop_neighbors:
        neighbors_of_neighbor = adjacency_matrix[neighbor].nonzero().squeeze(1).tolist()
        three_hop_neighbors.update(neighbors_of_neighbor)

    # Get 4-hop neighbors
    four_hop_neighbors = set()
    for neighbor in three_hop_neighbors:
        neighbors_of_neighbor = adjacency_matrix[neighbor].nonzero().squeeze(1).tolist()
        four_hop_neighbors.update(neighbors_of_neighbor)

    # Combine 1-hop, 2-hop, 3-hop and 4-hop neighbors and don't remove the original node /!\
    neighborhood = one_hop_neighbors.union(two_hop_neighbors).union(three_hop_neighbors).union(four_hop_neighbors)

    # check we don't remove the original node /!\
    #neighborhood.discard(node_id)

    neighborhood = list(neighborhood)

    return neighborhood

def get_neighbourhood_2(node_idx, adjacency_matrix, n_hops, features, labels):
    if adjacency_matrix.is_sparse:
        adjacency_matrix = adjacency_matrix.to_dense()

    if features.is_sparse:
        features = features.to_dense()
    # Number of nodes
    num_nodes = adjacency_matrix.shape[0]
    
    # Initialize a vector to keep track of reachable nodes
    reachable = torch.zeros(num_nodes, dtype=torch.bool)
    reachable[node_idx] = True
    
    # Use matrix multiplication to find nodes reachable within n_hops
    current_hop = torch.zeros(num_nodes, dtype=torch.bool)
    current_hop[node_idx] = True
    
    for _ in range(n_hops):
        # Find nodes reachable in the next hop
        next_hop = torch.matmul(adjacency_matrix, current_hop.float()).bool()
        # Update the reachable nodes
        reachable = reachable | next_hop
        current_hop = next_hop & (~reachable)
    
    # Extract the subgraph for the reachable nodes
    sub_adjacency_matrix = adjacency_matrix[reachable][:, reachable]
    
    # Extract features and labels for the reachable nodes
    sub_features = features[reachable]
    sub_labels = labels[reachable]
    
    # Create a new index mapping for the nodes in the subgraph
    new_index_mapping = torch.arange(num_nodes)[reachable]
    node_dict = {original_idx.item(): new_idx for new_idx, original_idx in enumerate(new_index_mapping)}
    
    return sub_adjacency_matrix, sub_features, sub_labels, node_dict

# def get_neighbourhood(node_idx, adjacency_matrix, n_hops, features, labels):
#     edge_index, _ = dense_to_sparse(adjacency_matrix)
#     edge_index = edge_index.long()
#     edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index) 
#     edge_subset_relabel = subgraph(edge_subset[0], edge_index, relabel_nodes=True)       # Get relabelled subset of edges
#     print("Edge subset: {}".format(edge_subset))
#     # edge_subset = torch.tensor(edge_subset, dtype=torch.long)

#     sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
#     print(sub_adj)
#     sub_feat = features.to_dense()[edge_subset[0], :]
#     sub_labels = labels[edge_subset[0]]
#     new_index = np.array([i for i in range(len(edge_subset[0]))])
#     node_dict = dict(zip(edge_subset[0].numpy(), new_index))     
	
   
#     print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
#     print("Num edges in subgraph counting self-loops: {}".format(sub_adj.sum()))
#     print("Num features in subgraph: {}".format(sub_feat.shape[0]))
	
#     return sub_adj, sub_feat, sub_labels, node_dict

def get_neighbourhood(node_idx, edge_index, n_hops, features, labels):
	edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index[0])     # Get all nodes involved
	edge_subset_relabel = subgraph(edge_subset[0], edge_index[0], relabel_nodes=True)       # Get relabelled subset of edges
	sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
	sub_feat = features[edge_subset[0], :]
	sub_labels = labels[edge_subset[0]]
	new_index = np.array([i for i in range(len(edge_subset[0]))])
	node_dict = dict(zip(edge_subset[0].numpy(), new_index))        # Maps orig labels to new
	# print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
	return sub_adj, sub_feat, sub_labels, node_dict

def extract_subgraph(neighborhood, adjacency_matrix, feature_matrix):
    # Convert the sparse adjacency matrix to dense if it's in sparse format
    if adjacency_matrix.is_sparse:
        adjacency_matrix = adjacency_matrix.to_dense()

    # Convert the sparse feature matrix to dense if it's in sparse format
    if feature_matrix.is_sparse:
        feature_matrix = feature_matrix.to_dense()

    # Ensure neighborhood is a list or convert it to a list
    if not isinstance(neighborhood, list):
        neighborhood = list(neighborhood)

    # Extract subgraph adjacency matrix
    A_v = adjacency_matrix[neighborhood][:, neighborhood]

    # Extract feature matrix for the subgraph
    X_v = feature_matrix[neighborhood]

    #test if the subgraph is connected
    print("Is the subgraph connected ?")
    print(csr_matrix(A_v))
    print(A_v.shape == (len(neighborhood), len(neighborhood)))
    print(X_v.shape[0] == len(neighborhood))

    return A_v, X_v

def extract_subgraph_2(neighborhood, adjacency_matrix, feature_matrix):
    # Convert the sparse adjacency matrix to dense if it's in sparse format
    if adjacency_matrix.is_sparse:
        adjacency_matrix = adjacency_matrix.to_dense()

    # Convert the sparse feature matrix to dense if it's in sparse format
    if feature_matrix.is_sparse:
        feature_matrix = feature_matrix.to_dense()

    # Ensure neighborhood is a list or convert it to a list
    if not isinstance(neighborhood, list):
        neighborhood = list(neighborhood)

    # Extract subgraph adjacency matrix
    A_v = adjacency_matrix[neighborhood][:, neighborhood]

    # Extract feature matrix for the subgraph
    X_v = feature_matrix[neighborhood]

    #node_dict to map the original node indices to new indices
    new_index = np.array([i for i in range(len(neighborhood))]).tolist()
    node_dict = dict(zip(neighborhood, new_index))

    #test if the subgraph is connected
    print("Is the subgraph connected ?")
    print(csr_matrix(A_v))
    print(A_v.shape == (len(neighborhood), len(neighborhood)))
    print(X_v.shape[0] == len(neighborhood))

    return A_v, X_v, node_dict


# Example usage
# neighborhood = get_2_hop_neighborhood(node_id, full_adjacency_matrix)
# A_v, X_v = extract_subgraph(neighborhood, full_adjacency_matrix, full_feature_matrix)



def load_node_specific_data(node_id, full_adjacency_matrix, full_feature_matrix, hops=2):
    """
    Load the subgraph data specific to a node based on its l-hop neighborhood.
    
    Parameters:
    - node_id: The ID of the target node.
    - full_adjacency_matrix: The full adjacency matrix of the graph.
    - full_feature_matrix: The full feature matrix of the graph.
    - hops: Number of hops to consider for neighborhood. Default is 2.
    
    Returns:
    - A_v: The adjacency matrix of the subgraph for the l-hop neighborhood.
    - X_v: The feature matrix of the nodes in the l-hop neighborhood.
    - x: The feature vector of the target node.
    """
    
    # Use Dijkstra's algorithm to find the shortest paths from node_id
    distances_from_node, _ = dijkstra(csgraph=full_adjacency_matrix, 
                                      directed=False, 
                                      indices=node_id, 
                                      return_predecessors=True)
    
    # Find nodes within 'hops' distance
    within_hops = np.where(distances_from_node <= hops)[0]
    
    # Extract the subgraph adjacency matrix
    A_v = full_adjacency_matrix[within_hops, :][:, within_hops]
    
    # Extract the subgraph feature matrix
    X_v = full_feature_matrix[within_hops, :]
    
    # Extract the feature vector for the target node
    x = full_feature_matrix[node_id, :]
    
    # Convert to torch tensors if not already
    A_v = torch.tensor(A_v.todense(), dtype=torch.float32)
    X_v = torch.tensor(X_v.todense(), dtype=torch.float32)
    x = torch.tensor(x.todense(), dtype=torch.float32)
    
    return A_v, X_v, x

# Usage example (assuming 'full_adj_matrix' and 'full_feat_matrix' are your full adjacency and feature matrices):
# node_id = 0  # Replace with the actual node ID
# A_v, X_v, x = load_node_specific_data(node_id, full_adj_matrix, full_feat_matrix)


# Usage example:
# predictions = load_model_predictions()
# node_id, predicted_class, confidence = select_node_by_confidence(predictions, threshold=0.5, highest=False)
# node_id, predicted_class, true_label = select_node_by_prediction_error(predictions, true_labels)
