import torch
import numpy as np
from torch.autograd import Variable
import copy
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.utils import dense_to_sparse
import time
import pickle
import sys
# sys.path.append('model')

# Import necessary modules from other scripts
from model.Data_loading import load_tree_dataset
from model.GCN import GCN, GCN3L, GCNSynthetic
from model.utils import normalize_adjacency_matrix, get_nodes_w_confidence, get_neighbourhood_fromRE
from model.cf_synthetic import CFExplainer
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

from collections import defaultdict
import torch
import numpy as np

def compute_variation_score(explanation1, explanation2):
    """
    Compute a score indicating the variation between two explanations.
    """
    diff = torch.abs(explanation1 - explanation2)
    variation_score = torch.sum(diff).item()
    return variation_score

def cons_stability_analysis(model, A, X, y_true, node_ids, num_iterations=10):
    """
    Assess the consistency of counterfactual explanations for a given set of nodes.
    """
    subgraph_sizes = {}
    explanation_variations = defaultdict(list)
    num_edges_removed = defaultdict(list)

    for node_id in node_ids:
        sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood_fromRE(int(node_id), A, 4, X, y_true)
        new_idx = node_dict[int(node_id)]
        subgraph_size = len(node_dict)
        subgraph_sizes[node_id] = subgraph_size
        explanation_sets = []

        for _ in range(num_iterations):
            print(f"Generating explanation for node {node_id} iteration {_ + 1}/{num_iterations}")
            CF_ = CFExplainer(model=model, sub_adj=sub_adj, sub_feat=sub_feat, n_hid=20,
                              dropout=0.0, sub_labels=sub_labels, y_pred_orig=y_true[new_idx],
                              num_classes=len(y_true.unique()), beta=0.5, device='cpu',
                              model_type='GCNSyntheticPerturb_synthetic')
            explanation = CF_.explain(node_idx=node_id, cf_optimizer='SGD', new_idx=new_idx, lr=0.1,
                                      n_momentum=0.0, num_epochs=500)
            explanation_sets.append(explanation[2])  # Assuming the explanation adjacency matrix is at index 2
            num_edges_removed[node_id].append(torch.sum(sub_adj - explanation[2]).item() / 2)

        # Compute variation in explanations
        for i in range(len(explanation_sets) - 1):
            for j in range(i + 1, len(explanation_sets)):
                variation_score = compute_variation_score(explanation_sets[i], explanation_sets[j])
                explanation_variations[node_id].append(variation_score)

    return subgraph_sizes, dict(explanation_variations), dict(num_edges_removed)


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
    print(len(sizes))
    print(len(variations))

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
    torch.autograd.set_detect_anomaly(True)
    
    #------------------------------------------------
    # Test on tree dataset
    #------------------------------------------------
    
    # Load the tree dataset return adj, features, labels, idx_train, idx_test
    A, X, y_true, idx_train, idx_test = load_tree_dataset("data/")
    # A, X, y_true, idx_train, idx_test = load_BA_dataset("data/")
    edge_index = dense_to_sparse(A)
    norm_adj = normalize_adjacency_matrix(A)
    norm_edge_index = dense_to_sparse(norm_adj)
    print(norm_edge_index)
    print(norm_edge_index[0])

    print(norm_adj)
    print(norm_adj[0])


    print(A.shape)
    print(X.shape)
    print(y_true.shape)
    print(idx_train.shape)
    print(idx_test.shape)
    print(len(norm_adj))
    print("___________________")
    print(idx_test)
    time.sleep(10)

    # Define the GCN model
    #for tree shape dataset
    # model = GCN3L(nfeat=X.shape[1], nhid=20, nout=20, nclass=len(np.unique(y_true)), dropout=0.0)
    model = GCNSynthetic(nfeat=X.shape[1], nhid=20, nout=20,
						 nclass=len(y_true.unique()), dropout=0.1)

    #for BA dataset
    # model = GCN3L(nfeat=X.shape[1], nhid=16, nout=16, nclass=len(np.unique(y_true)), dropout=0.9)


    # Load the state_dict from file
    state_dict = torch.load("data/model_3layer/gcn_3layer_syn4.pt")
    #state_dict = torch.load("data/model_3layer/gcn_3layer_cora.pt")

    print(f"KEYS of our state dictionary", state_dict.keys())

    try:
        model.load_state_dict(state_dict)
    except:
        # Create a new state_dict with the correct keys
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # Adjust keys for graph convolutional layers
            if 'weight' in key and 'gc' in key:
                new_key = key.replace('weight', 'W')
            # Adjust keys for the linear layer if needed
            # Note: This assumes the linear layer uses 'weight' and 'bias' as in the loaded state_dict
            # If the model expects different keys for the linear layer, adjust accordingly
            elif 'lin' in key:
                # No changes needed if your linear layer also uses 'weight' and 'bias'
                # If your model uses different key names for the linear layer, adjust them here
                pass
            # Optionally, skip 'bias' keys if your model does not use them for graph convolutional layers
            elif 'bias' in key and 'gc' in key:
                continue
            new_state_dict[new_key] = value

        # Now load the modified state_dict into your model
        model.load_state_dict(new_state_dict)
    """ 
    # Create a new state_dict with the correct keys
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # Adjust keys for graph convolutional layers
        if 'weight' in key and 'gc' in key:
            new_key = key.replace('weight', 'W')
        # Adjust keys for the linear layer if needed
        # Note: This assumes the linear layer uses 'weight' and 'bias' as in the loaded state_dict
        # If the model expects different keys for the linear layer, adjust accordingly
        elif 'lin' in key:
            # No changes needed if your linear layer also uses 'weight' and 'bias'
            # If your model uses different key names for the linear layer, adjust them here
            pass
        # Optionally, skip 'bias' keys if your model does not use them for graph convolutional layers
        elif 'bias' in key and 'gc' in key:
            continue
        new_state_dict[new_key] = value 

    # Now load the modified state_dict into your model
    model.load_state_dict(new_state_dict) """

    model.eval()
    norm_A = normalize_adjacency_matrix(A.to_dense())
    model_predictions = model(X, norm_A)
    y_pred_orig = torch.argmax(model_predictions, dim=1)
    print(f"Original predictions: {y_pred_orig}")

    #CF_Explain5
    data = []  # Initialize an empty list to collect data
    examples = []

    # idx_test = idx_test[:]
    # perform counterfactual explanation for all the nodes in the test set
    for node_id in idx_test:
        print(f"Generating counterfactual explanation for node {node_id}")
        print(f"X: {X}")
        print(f"y_true: {y_true}")
        sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood_fromRE(int(node_id), A, 4, X, y_true)

        print(node_dict)
        new_idx = node_dict[int(node_id)]
        print(f"New index: {new_idx}")
	    
        if sub_adj.nelement() == 0:
            print("EUREKA")
            sub_adj = torch.Tensor(0)
            print(sub_adj)

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
							n_hid=20,
							dropout=0.0,
							sub_labels=sub_labels,
							y_pred_orig=model_predictions[node_id],
							num_classes = len(y_true.unique()),
							beta=0.5,
							device='cpu',
							model_type='GCNSyntheticPerturb_synthetic')
        
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

    with open("data/cf_examples_synthetic.pickle", "wb") as f:
        pickle.dump(examples, f)
    #save the dataset
    df = pd.DataFrame(data)  # Convert list of dicts to DataFrame
    df.to_pickle("data/cf_explanations_synthetic.pickle")
    
    print(df) 

    #Calculate the metrics on the dataset
    """
    We generate a CF example for each node in the graph separately and evaluate in terms of four metrics.

    Fidelity: is defined as the proportion of nodes where the original predictions match the prediction for the explanations (Molnar 2019 Ribeiro et al. 2016). Since we generate \(\mathrm{CF}\) examples, we do not want the original prediction to match the prediction for the explanation, so we want a low value for fidelity.

    Explanation Size: is the number of removed edges. It corresponds to the \(\mathcal{L}_{\text {dist }}\) term in Equation 1 the difference between the original \(A_{v}\) and the counterfactual \(\overline{A_{v}}\). Since we want to have minimal explanations, we want a small value for this metric. Note that we cannot evaluate this metric for GNNEXPLAINER.

    Sparsity: measures the proportion of edges in \(A_{v}\) that are removed (Yuan et al. 2020b). A value of 0 indicates all edges in \(A_{v}\) were removed. Since we want minimal explanations, we want a value close to 1 . Note that we cannot evaluate this metric for GNNEXPLAINER.

    Accuracy: is the mean proportion of explanations that are "correct". Following Ying et al. (2019) and Luo et al. (2020), we only compute accuracy for nodes that are originally predicted as being part of the motifs, since accuracy can only be computed on instances for which we know the ground truth explanations. Given that we want minimal explanations, we consider an explanation to be correct if it exclusively involves edges that are inside the motifs (i.e., only removes edges that are within the motifs).
    
    """    
    """ # Plot histogram showing the distribution of CF examples by the number of edges removed, normalized to show proportions
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(df["num_edges_removed"], bins=20, density=True)  # Use density=True for proportions
    ax.set_title("Distribution of Explanation Size")
    ax.set_xlabel("Number of edges removed")
    ax.set_ylabel("Proportion of CF examples")
    ax.set_xlim(0, 20)
    #between 0 and 20 set ticks every 2
    ax.set_xticks(np.arange(0, 21, 2))
    plt.savefig("explanation_size_sparsity.png")
    plt.show() """



    # Load the CF examples
    header = ["node_idx", "new_idx", "cf_adj", "sub_adj", "y_pred_orig", "y_pred_new", "y_pred_new_actual",
            "label", "num_nodes", "loss_total", "loss_pred", "loss_graph_dist"]
    
    with open("data/cf_examples_synthetic.pickle", "rb") as f:
        cf_examples = pickle.load(f)
        df_prep = []
        for example in cf_examples:
            if example != []:
                df_prep.append(example)
        df = pd.DataFrame(df_prep, columns=header)

    print(df)

    # Add num edges
    num_edges = []
    for i in df.index:
        num_edges.append(sum(sum(df["sub_adj"][i])) / 2)
    df["num_edges"] = [tensor.item() for tensor in num_edges]

    #devide by two all the 'num_edges_removed' values
    df["loss_graph_dist"] = df["loss_graph_dist"] / 2

    print(df)

    print(f"y_pred_orig", df["y_pred_orig"].unique())

    # For accuracy, only look at motif nodes
    df_motif = df[df["y_pred_orig"] != 0].reset_index(drop=True)
    accuracy = []

    print(f"DF motif", df_motif)
    #df motif node_idx
    print(f"DF motif node_idx", df_motif["node_idx"])

    # Get original predictions
    dict_ypred_orig = dict(zip(sorted(np.concatenate((idx_train.numpy(), idx_test.numpy()))), y_pred_orig.numpy()))

    for i in range(len(df_motif)):
        print(i)
        node_idx = df_motif["node_idx"][i].item()
        new_idx = df_motif["new_idx"][i]
        _, _, _, node_dict = get_neighbourhood_fromRE(int(node_idx), A, 4, X, y_true)

        # Confirm idx mapping is correct
        if node_dict[node_idx] == df_motif["new_idx"][i]:

            cf_adj = df_motif["cf_adj"][i]
            sub_adj = df_motif["sub_adj"][i]
            #require gradient false
            cf_adj.requires_grad = False
            sub_adj.requires_grad = False
            perturb = np.abs(cf_adj - sub_adj)
            perturb_edges = np.nonzero(perturb)  # Edge indices

            nodes_involved = np.unique(np.concatenate((perturb_edges[0], perturb_edges[1]), axis=0))
            perturb_nodes = nodes_involved[nodes_involved != new_idx]  # Remove original node

            # Retrieve original node idxs for original predictions
            perturb_nodes_orig_idx = []
            for j in perturb_nodes:
                perturb_nodes_orig_idx.append([key for (key, value) in node_dict.items() if value == j])
            perturb_nodes_orig_idx = np.array(perturb_nodes_orig_idx).flatten()

            # Retrieve original predictions
            perturb_nodes_orig_ypred = np.array([dict_ypred_orig[k] for k in perturb_nodes_orig_idx])
            nodes_in_motif = perturb_nodes_orig_ypred[perturb_nodes_orig_ypred != 0]
            prop_correct = len(nodes_in_motif) / len(perturb_nodes_orig_idx)

            accuracy.append([node_idx, new_idx, perturb_nodes_orig_idx,
                            perturb_nodes_orig_ypred, nodes_in_motif, prop_correct])

    df_accuracy = pd.DataFrame(accuracy, columns=["node_idx", "new_idx", "perturb_nodes_orig_idx",
                                                "perturb_nodes_orig_ypred", "nodes_in_motif", "prop_correct"])

    print(f"DF bails", df_accuracy)

    # Print the results
    print("Num cf examples found: {}/{}".format(len(df), len(idx_test)))
    print("Avg fidelity: {}".format(1 - len(df) / len(idx_test)))
    print("Average graph distance: {}, std: {}".format(np.mean(df["loss_graph_dist"]), np.std(df["loss_graph_dist"])))
    print("Average sparsity: {}, std: {}".format(np.mean(1 - df["loss_graph_dist"] / df["num_edges"]), np.std(1 - df["loss_graph_dist"] / df["num_edges"])))
    print("Accuracy", np.mean(df_accuracy["prop_correct"]), np.std(df_accuracy["prop_correct"]))
    print(" ")
    print("***************************************************************")
    print(" ")
    time.sleep(15)

    """ 
    # Stability analysis
    # Define the set of node IDs for which to generate explanations
    high_confidence_nodes, low_confidence_nodes = get_nodes_w_confidence(model_predictions, y_true, A, threshold=0.9)

    #select a subset of nodes
    high_confidence_nodes = high_confidence_nodes[:5]
    print(f"High confidence nodes: {high_confidence_nodes}")

    # Assess the consistency of counterfactual explanations
    subgraph_sizes, explanation_variations, num_edges_removed = cons_stability_analysis(model, A, X, y_true, high_confidence_nodes, num_iterations=10)

    print(f"Subgraph sizes: {subgraph_sizes}")
    print(f"Explanation variations: {explanation_variations}")
    print(f"Number of edges removed: {num_edges_removed}")
    time.sleep(10)
    # Plot the relationship between subgraph sizes, explanation stability scores, and the number of edges removed
    plot_stability_vs_subgraph_size(subgraph_sizes, explanation_variations, num_edges_removed) """

if __name__ == "__main__":
    main()
