import torch
import numpy as np
from torch.autograd import Variable
import copy
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import sys
sys.path.append('model')

# Import necessary modules from other scripts
from model.GCN import GCN
from model.Model import compute_loss, compute_accuracy
from model.utils import get_2_hop_neighborhood, extract_subgraph, normalize_adjacency_matrix, get_neighbourhood_fromRE
from model.cf_cora import CFExplainer
from collections import defaultdict

def compute_jaccard_similarity(set1, set2):
    """
    Compute the Jaccard similarity between two sets.
    """
    intersection = torch.sum(set1 & set2).float()
    union = torch.sum(set1 | set2).float()
    return intersection / union if union > 0 else 0

def perturb_model(model, noise_level=0.01):
    """
    Perturb the model weights slightly by adding Gaussian noise.
    """
    new_model = copy.deepcopy(model)
    for param in new_model.parameters():
        param.data += noise_level * torch.randn_like(param)
    return new_model

def perturb_data(A, perturbation_rate=0.01):
    """
    Randomly add or remove edges from the adjacency matrix A to simulate data perturbation.
    """
    perturbed_A = A.clone().to_dense()
    num_edges = perturbed_A.nonzero(as_tuple=False).size(0)
    num_perturbations = int(perturbation_rate * num_edges)

    for _ in range(num_perturbations):
        i = np.random.randint(0, perturbed_A.size(0))
        j = np.random.randint(0, perturbed_A.size(1))
        perturbed_A[i, j] = 1 - perturbed_A[i, j]  # Flip the edge
        perturbed_A[j, i] = 1 - perturbed_A[j, i]  # Ensure symmetry for undirected graphs
    
    return perturbed_A

def cons_stability_analysis(model, A, X, y_true, node_ids, num_iterations=10):
    """
    Assess the consistency of counterfactual explanations for a given set of nodes without perturbing the model or adjacency matrix.

    Args:
        model: The GCN model used for generating explanations.
        A: Adjacency matrix of the graph.
        X: Feature matrix of the nodes.
        y_true: True labels of the nodes.
        node_ids: A set of node IDs for which to generate explanations.
        num_iterations: Number of times to compute the explanation for each node.

    Returns:
        A tuple containing dictionaries for subgraph sizes, explanation variation scores, and number of edges removed.
    """
    subgraph_sizes = {}
    explanation_variations = defaultdict(list)
    num_edges_removed = defaultdict(list)

    for node_id in node_ids:
        original_neighborhood = get_2_hop_neighborhood(node_id, A)
        subgraph_size = len(original_neighborhood)
        subgraph_sizes[node_id] = subgraph_size
        explanation_sets = []

        A_v, X_v = extract_subgraph(original_neighborhood, A, X)
        if not isinstance(original_neighborhood, list):
            original_neighborhood = list(original_neighborhood)
        y_true_sub = y_true[original_neighborhood]

        for _ in range(num_iterations):
            print(f"Generating explanation for node {node_id} iteration {_ + 1}/{num_iterations}")
            CF_ = CFExplainer(model, X_v, A_v, y_true_sub)
            explanation = CF_.generate_explanation()
            explanation_sets.append(explanation)
            num_edges_removed[node_id].append(torch.sum(A_v - explanation))

        # Compute variation in explanations
        for i in range(len(explanation_sets) - 1):
            for j in range(i + 1, len(explanation_sets)):
                variation_score = compute_variation_score(explanation_sets[i], explanation_sets[j])
                explanation_variations[node_id].append(variation_score)

    return subgraph_sizes, explanation_variations, num_edges_removed

def compute_variation_score(explanation1, explanation2):
    """
    Compute a score indicating the variation between two explanations.

    Args:
        explanation1: The adjacency matrix of the first explanation subgraph.
        explanation2: The adjacency matrix of the second explanation subgraph.

    Returns:
        A score representing the difference between the two explanations.
    """
    diff = torch.abs(explanation1 - explanation2)
    variation_score = torch.sum(diff).item()
    return variation_score

def plot_stability_vs_subgraph_size(subgraph_sizes, explanation_variations, num_edges_removed):
    """
    Plot the relationship between subgraph sizes, explanation stability scores, and the number of edges removed.

    Args:
        subgraph_sizes: Dictionary mapping node ID to its subgraph size.
        explanation_variations: Dictionary mapping node ID to its explanation variation scores.
        num_edges_removed: Dictionary mapping node ID to the number of edges removed in each explanation.
    """
    avg_variation_scores = {node_id: np.mean(scores) for node_id, scores in explanation_variations.items()}
    avg_num_edges_removed = {node_id: np.mean(counts) for node_id, counts in num_edges_removed.items()}

    sizes = list(subgraph_sizes.values())
    variations = list(avg_variation_scores.values())
    edges_removed = list(avg_num_edges_removed.values())

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Subgraph Size')
    ax1.set_ylabel('Avg. Explanation Variation Score', color=color)
    ax1.scatter(sizes, variations, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Avg. Number of Edges Removed', color=color)
    ax2.scatter(sizes, edges_removed, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Explanation Stability vs Subgraph Size')
    #save the plot
    plt.savefig('stability_vs_subgraph_size.png')
    plt.show()

def main():
    """
    Main file to generate counterfactual explanations for a node in the Cora dataset.
    Two use cases are demonstrated:
    I. Counterfactual explanation for fixed node on a fixed model and data
        1. Counterfactual explanation for a confident prediction
        2. Counterfactual explanation for an uncertain prediction
    II. Stability analysis of counterfactual explanations
        According to the paper, the counterfactual explanations should be stable with respect to the model and the data.
    """

    # I. a) Load the model and the data based on .pt files
    
    #Load the model and the data
    A = torch.load('data/model_2layer/full_adjacency_matrix.pt')
    X = torch.load('data/model_2layer/feature_matrix.pt')
    y_true = torch.load('data/model_2layer/true_labels.pt')
    y_pred = torch.load('data/model_2layer/model_predictions.pt')
    W = torch.load('data/model_2layer/weight_matrix.pt')
    train_idx = torch.load('data/model_2layer/train_idx.pt')
    test_idx = torch.load('data/model_2layer/test_idx.pt')

    #define GCN model
    model = GCN(X.shape[1], 32, len(np.unique(y_true)))
    model.load_state_dict(torch.load('data/model_2layer/model.pt'))
    model.eval()

    A_hat = normalize_adjacency_matrix(A.to_dense())
    A_without_self_loops = A_hat - torch.eye(A_hat.size(0))

    model_predictions = model(X, A_hat)
    test_loss = compute_loss(model_predictions[test_idx], y_true[test_idx])
    test_acc = compute_accuracy(model_predictions.argmax(dim=1)[test_idx], y_true[test_idx])
    print(f'CHECK Test Loss: {test_loss.item():0.4f}, Test Acc: {test_acc:0.4f}')
    y_pred_orig = torch.argmax(model_predictions, dim=1)
    print(f"Original predictions: {y_pred_orig}")

    #CF_Explain5
    data = []  # Initialize an empty list to collect data
    examples = []  # Initialize an empty list to collect data

    """ idx_test = []
    #selecting only the nodes with more than 1 neighbour
    for node_id in test_idx:
        sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood_fromRE(int(node_id), A.to_dense(), 3, X, y_true)
        if len(node_dict) > 1:
            idx_test.append(node_id)
    
    #save the idx_test
    with open("data/idx_test.pickle", "wb") as f:
        pickle.dump(idx_test, f) """

    #load the idx_test
    with open("data/idx_test.pickle", "rb") as f:
        idx_test = pickle.load(f)

    for node_id in idx_test:
        print(f"Generating counterfactual explanation for node {node_id}")
        print(f"X: {X}")
        print(f"y_true: {y_true}")
        sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood_fromRE(int(node_id), A_without_self_loops.to_dense(), 3, X, y_true)

        print(node_dict)
        new_idx = node_dict[int(node_id)]
        print(f"New index: {new_idx}")
	    
        if sub_adj.nelement() == 0:
            print("EUREKA")
            sub_adj = torch.Tensor(0)
            print(sub_adj)

        #deleting diagonal elements
        sub_adj = sub_adj - torch.diag(sub_adj.diag())

        print(f"Subgraph adjacency matrix: {sub_adj}")
        print(f"Subgraph feature matrix: {sub_feat}")
        print(f"Subgraph labels: {sub_labels}")

        # time.sleep(5)

        # Check that original model gives same prediction on full graph and subgraph
        with torch.no_grad():
            print(f"Original prediction for node {node_id}: {model_predictions[node_id]}")
            
        CF_ = CFExplainer(model=model,
							sub_adj=sub_adj,
							sub_feat=sub_feat,
							n_hid=32,
							dropout=0.5,
							sub_labels=sub_labels,
							y_pred_orig=model_predictions[node_id],
							num_classes = len(y_true.unique()),
							beta=0.5,
							device='cpu',
							model_type='GCN')
        
        cf_example = CF_.explain(node_idx=node_id,
                                  cf_optimizer='SGD', new_idx=new_idx,
                                    lr=0.1,
	                               n_momentum=0.0, num_epochs=500)
        print(cf_example)
        
        for ex in cf_example:
            # print(ex[3])
            print(ex[9])
            print(ex[11])

        if cf_example != []:
            examples.append(cf_example[0])
            best_ex = cf_example[0]
            print(f'Counterfactual explanation for node {node_id}: {best_ex}')
            num_edges_removed = torch.sum(sub_adj - best_ex[2]).item() /2
            data.append({"node_idx": node_id, "X_v": sub_feat, "A_v": sub_adj, "Cf" : best_ex[2], "num_edges_removed": num_edges_removed})

    with open("data/cf_examples_cora.pickle", "wb") as f:
        pickle.dump(examples, f)
    #save the dataset
    df = pd.DataFrame(data)  # Convert list of dicts to DataFrame
    df.to_pickle("data/cf_explanations_cora.pickle")
    
    print(df)

    
if __name__ == "__main__":
    main()
