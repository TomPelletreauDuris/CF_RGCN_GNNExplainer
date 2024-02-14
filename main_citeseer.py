import torch
import numpy as np
from torch.autograd import Variable
import copy
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import time
import pickle
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F
import sys
# sys.path.append('model')

# Import necessary modules from other scripts

from model.Data_loading import load_cora_dataset, load_tree_dataset, load_BA_dataset
from model.GCN import GCN, GCN3L, GCNSynthetic, GCN3Layer_PyG, RGCN
from model.Model import compute_loss, compute_accuracy, get_the_model
from model.utils import select_node_by_confidence, get_2_hop_neighborhood,get_4_hop_neighborhood, extract_subgraph,extract_subgraph_2, normalize_adjacency_matrix, get_nodes_w_confidence, get_neighbourhood, get_neighbourhood_2, get_neighbourhood_fromRE, get_neighbourhood_fromRGCN
from model.cf_citeseer import CFExplainer
# from model.cf_2 import CF_GNNExplainer
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
            # CF_ = CF_GNNExplainer(model, X_v, A_v, y_true_sub)
            # explanation = CF_.generate_explanation()
            # explanation_sets.append(explanation)
            # num_edges_removed[node_id].append(torch.sum(A_v - explanation))

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
    
    transform = T.Compose([T.ToUndirected(), T.NormalizeFeatures()])
    dataset = Planetoid(root='data', name='CiteSeer', transform=transform)
    
    adjMatrix = [[0 for i in range(len(dataset.data.y))] for k in range(len(dataset.data.y))]

    # scan the arrays edge_u and edge_v
    for i in range(len(dataset.data.edge_index[0])):
        u = dataset.data.edge_index[0][i]
        v = dataset.data.edge_index[1][i]
        adjMatrix[u][v] = 1


    # For models trained using our GCN_synethic from GNNExplainer,
    # using hyperparams from GNN explainer tasks
    adj = torch.Tensor(adjMatrix).squeeze()
    features = torch.Tensor(dataset.data.x).squeeze()
    labels = torch.tensor(dataset.data.y).squeeze()

    node_idx = [i for i in range(0, len(dataset.data.y))]
    idx_train = torch.masked_select(torch.Tensor(node_idx), dataset.data.train_mask)
    idx_test = torch.masked_select(torch.Tensor(node_idx), dataset.data.test_mask)
    idx_train = idx_train.type(torch.int64)
    idx_test = idx_test.type(torch.int64)
    
    norm_edge_index = dense_to_sparse(adj)
    print(norm_edge_index)
    print(norm_edge_index[0])


    print(adj.shape)
    print(features.shape)
    print(labels.shape)
    print(idx_train.shape)
    print(idx_test.shape)
    print(len(norm_edge_index))
    print("___________________")

    norm_adj = normalize_adjacency_matrix(adj.to_sparse().to_dense())
    norm_edge_index = dense_to_sparse(norm_adj)
    print(norm_edge_index)

    model = GCN3Layer_PyG(nfeat=features.shape[1], nhid=64, nout=64,
					 nclass=len(labels.unique()), dropout=0.05)
    
    # Load the state_dict from file
    state_dict = torch.load("data/model_3layer/GCN3Layer_PyG_citeseer.pt")
    model.load_state_dict(state_dict)
    model.eval()

    # Convert a sparse adjacency matrix to an edge index list
    sparse_adj = adj.to_sparse()
    edge_index_adj = sparse_adj.coalesce().indices()
    # sparse_adj = torch.sparse_coo_tensor(dataset.data.edge_index, torch.ones(dataset.data.edge_index.shape[1]), (len(dataset.data.y), len(dataset.data.y)))

    # print(sparse_adj.shape)
    # print(edge_index_adj.shape)
    # print(dataset.data.edge_index.shape)
    # print(adj.shape)


    model_predictions = model(features, edge_index_adj)

    # output = model(features, norm_adj)
    y_pred_orig = torch.argmax(model_predictions, dim=1)
    
    idx_test = torch.Tensor([2333, 2266, 2560, 1797, 2706, 2684, 2443, 2177, 1788,
                            1825, 2353, 1888, 2494, 2305, 1710, 2189, 2071,
            2606, 2038, 1725, 1800, 2693, 2114, 1894,
            1930, 2299, 2212, 1829, 2084, 2222, 2155, 2178, 1966, 2478, 1915, 2179,
            2093, 2021, 2304, 1782, 1813, 2651, 2621, 2292, 2059, 1970, 2057, 1738,
            1721, 2640, 1997, 1892, 1895, 2461, 1992, 2149, 2425, 2151, 2573, 2269,
            1998, 2642, 1819, 1843, 2337, 2232, 1855, 2167, 2325, 2349, 1823, 2062,
            1913, 1914, 2165, 2007, 2100, 2224, 2508, 2563, 2009, 2466, 2340, 2441,
            2519, 2491, 1812, 1961, 2538, 2502, 2378, 2459, 2589, 2704, 1799, 2024,
            1882, 2499, 1974, 1948])
    idx_test = idx_test.type(torch.int64)

    #----------------------


    data = []  # Initialize an empty list to collect data
    examples = []  # Initialize an empty list to collect counterfactual explanations

    """ for i in idx_test:
        print(f"Node {i} prediction: {y_pred_orig[i]}")
        sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood_fromRE(int(i), adj, 4, features, labels)
        #if nr edges in subgraph is 0, continue
        if sub_adj.shape[0] == 0:
            continue
        if (sub_adj.sum() / 2) == 0:
            continue

        new_idx = node_dict[int(i)]
        sub_sparse_adj = sub_adj.to_sparse()
        print(f"is the matrix symmetric: {torch.all(sub_adj.t() == sub_adj)}")
        sub_edge_index_adj = sub_sparse_adj.coalesce().indices()
        # sub_edge_index_adj = torch.stack([node_dict[edge[0].item()] for edge in sub_edge_index_adj.t()])
        print(f"context sub adj: {sub_adj.shape}")
        print(f'control sub features: {sub_feat.shape}')

        print(f"Original prediction for node {i}: {model_predictions[i]}")
        print(f"Prediction label for node {i}: {model_predictions[i].argmax().item()}")
        print(f"New index: {new_idx}")
        with torch.no_grad():
            sub_norm_edge_index = dense_to_sparse(sub_adj)
            print(sub_norm_edge_index)
            sub_model_predictions = model(sub_feat, sub_edge_index_adj)
        print(f"Prediction for subgraph: {sub_model_predictions[new_idx]}")
        print(f"Prediction label for node {i}: {sub_model_predictions[new_idx].argmax().item()}")
        # time.sleep(5)
        CF_ = CFExplainer(model=model,
                                sub_adj=sub_adj,
                                sub_feat=sub_feat,
                                n_hid=64,
                                dropout=0.00,
                                sub_labels=sub_labels,
                                y_pred_orig=y_pred_orig[i],
                                num_classes = len(labels.unique()),
                                beta=0.5,
                                device='cpu',
                                model_type='synthetic')
            
        cf_example = CF_.explain(node_idx=i,
                                    cf_optimizer='SGD', new_idx=new_idx,
                                        lr=0.2,
                                    n_momentum=0.0, num_epochs=500)
        print(f'l EXEMPLE', cf_example)
        if cf_example != []:
            examples.append(cf_example[0])
            print(f'Counterfactual explanation for node {i}: {cf_example[0][2]}')
            print(f'adjacency matrix: {sub_adj}')
            print(f"counterfactual explanation: {cf_example[0][3]}")
            print(f'number of edges removed: {torch.sum(sub_adj - cf_example[0][2]).item()}')
            # time.sleep(10)
    
            data.append({"node_idx": i, "X_v": sub_feat, "A_v": sub_adj, "num_edges_removed": torch.sum(sub_adj - cf_example[0][2]).item()/2})
        else:
            print(f'No counterfactual explanation for node {i}')
            data.append({"node_idx": i, "X_v": sub_feat, "A_v": sub_adj, "num_edges_removed": 0})
    
    with open("data/cf_examples_citeseer.pickle", "wb") as f:
        pickle.dump(examples, f)
    df = pd.DataFrame(data)  # Convert list of dicts to DataFrame
    df.to_pickle("data/cf_explanations_citeseer.pickle")
    
    #print average number of edges removed
    print(f"Average number of edges removed: {df['num_edges_removed'].mean()}")
    """

    #plot Histograms showing the proportion of CF examples for the number of edges removed 
    
    # Plot histogram showing the distribution of CF examples by the number of edges removed, normalized to show proportions
    #open df.to_pickle("data/cf_explanations_citeseer.pickle")
    df = pd.read_pickle("data/cf_explanations_citeseer.pickle")

    #get rid of the 0 values
    df = df[df["num_edges_removed"] != 0]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(df["num_edges_removed"], bins=20, density=True)  # Use density=True for proportions
    ax.set_title("Distribution of Explanation Size")
    ax.set_xlabel("Number of edges removed")
    ax.set_ylabel("Proportion of CF examples")
    ax.set_xlim(0, 20)
    #between 0 and 20 set ticks every 2
    ax.set_xticks(np.arange(0, 21, 2))
    plt.savefig("explanation_size_sparsity_fullCF_SGD_D005_lr05_b055_mm00.png")
    # plt.show() 
    

    # Load the CF examples
    header = ["node_idx", "new_idx", "cf_adj", "sub_adj", "y_pred_orig", "y_pred_new", "y_pred_new_actual",
            "label", "num_nodes", "loss_total", "loss_pred", "loss_graph_dist"]
    
    with open("data/cf_examples_citeseer.pickle", "rb") as f:
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

    #transform tensor to float
    df["loss_graph_dist"] = df["loss_graph_dist"].astype(float)

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

    """ for i in range(len(df_motif)):
        print(i)
        node_idx = df_motif["node_idx"][i].item()
        new_idx = df_motif["new_idx"][i]
        _, _, _, node_dict = get_neighbourhood_fromRE(int(node_idx), adj, 4, features, labels)

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

            print(f"perturb_nodes_orig_idx", perturb_nodes_orig_idx)
            print(f"dict_ypred_orig", dict_ypred_orig)

            # Retrieve original predictions
            perturb_nodes_orig_ypred = np.array([dict_ypred_orig[k] for k in perturb_nodes_orig_idx])
            nodes_in_motif = perturb_nodes_orig_ypred[perturb_nodes_orig_ypred != 0]
            prop_correct = len(nodes_in_motif) / len(perturb_nodes_orig_idx)

            accuracy.append([node_idx, new_idx, perturb_nodes_orig_idx,
                            perturb_nodes_orig_ypred, nodes_in_motif, prop_correct])
    """
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


















    #stop here the computation or perform a hyperparameter search













    exit()
    
    
    # Load the Cora dataset
    dropout_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.4]
    beta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    lr_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    n_momentum_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9]
    num_epochs_values = [10, 50, 100, 200, 300, 400, 500, 1000]

    #dropout
    for dropout in dropout_values:
        print(f"Dropout: {dropout}")
        model = GCN3Layer_PyG(nfeat=features.shape[1], nhid=64, nout=64,
                     nclass=len(labels.unique()), dropout=dropout)
        state_dict = torch.load("data/model_3layer/GCN3Layer_PyG_citeseer.pt")
        model.load_state_dict(state_dict)
        model.eval()
        data = []
        for i in idx_test[0:10]:
            print(f"Node {i} prediction: {y_pred_orig[i]}")
            sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood_fromRE(int(i), adj, 2, features, labels)
            new_idx = node_dict[int(i)]
            sub_sparse_adj = sub_adj.to_sparse()
            sub_edge_index_adj = sub_sparse_adj.coalesce().indices()
            print(f"context sub adj: {sub_adj.shape}")
            print(f'control sub features: {sub_feat.shape}')

            print(f"Original prediction for node {i}: {model_predictions[i]}")
            print(f"Prediction label for node {i}: {model_predictions[i].argmax().item()}")
            print(f"New index: {new_idx}")
            with torch.no_grad():
                sub_norm_edge_index = dense_to_sparse(sub_adj)
                print(sub_norm_edge_index)
                sub_model_predictions = model(sub_feat, sub_edge_index_adj)
            print(f"Prediction for subgraph: {sub_model_predictions[new_idx]}")
            print(f"Prediction label for node {i}: {sub_model_predictions[new_idx].argmax().item()}")
            # time.sleep(5)
            CF_ = CFExplainer(model=model,
                                    sub_adj=sub_adj,
                                    sub_feat=sub_feat,
                                    n_hid=64,
                                    dropout=dropout,
                                    sub_labels=sub_labels,
                                    y_pred_orig=y_pred_orig[i],
                                    num_classes = len(labels.unique()),
                                    beta=0.5,
                                    device='cpu',
                                    model_type='synthetic')
                
            cf_example = CF_.explain(node_idx=i,
                                        cf_optimizer='SGD', new_idx=new_idx,
                                            lr=0.1,
                                        n_momentum=0.9, num_epochs=500)
            print(f'l EXEMPLE', cf_example)
            if cf_example != []:
                print(f'Counterfactual explanation for node {i}: {cf_example[0][2]}')
                print(f'adjacency matrix: {sub_adj}')
                print(f"counterfactual explanation: {cf_example[0][3]}")
                print(f'number of edges removed: {torch.sum(sub_adj - cf_example[0][2]).item()}')
                # time.sleep(10)
                data.append({"node_idx": i, "X_v": sub_feat, "A_v": sub_adj, "num_edges_removed": torch.sum(sub_adj - cf_example[0][2]).item()})
            else:
                print(f'No counterfactual explanation for node {i}')
                data.append({"node_idx": i, "X_v": sub_feat, "A_v": sub_adj, "num_edges_removed": 0})

        df = pd.DataFrame(data)  # Convert list of dicts to DataFrame
        df.to_pickle(f"data/cf_explanations_dropout{dropout}.pickle")

        #print average number of edges removed
        print(f"Average number of edges removed: {df['num_edges_removed'].mean()}")
        # Plot histogram showing the distribution of CF examples by the number of edges removed, normalized to show proportions
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.hist(df["num_edges_removed"], bins=20, density=True)  # Use density=True for proportions
        ax.set_title(f"Distribution of Explanation Size for dropout {dropout}")
        ax.set_xlabel("Number of edges removed")
        ax.set_ylabel("Proportion of CF examples")
        ax.set_xlim(0, 20)
        #between 0 and 20 set ticks every 2
        ax.set_xticks(np.arange(0, 21, 2))
        plt.savefig(f"explanation_size_sparsity_dropout{dropout}.png")
    
    #beta
    for beta in beta_values:
        print(f"Beta: {beta}")
        model = GCN3Layer_PyG(nfeat=features.shape[1], nhid=64, nout=64,
                     nclass=len(labels.unique()), dropout=0.01)
        state_dict = torch.load("data/model_3layer/GCN3Layer_PyG_citeseer.pt")
        model.load_state_dict(state_dict)
        model.eval()
        data = []
        for i in idx_test[0:10]:
            print(f"Node {i} prediction: {y_pred_orig[i]}")
            sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood_fromRE(int(i), adj, 2, features, labels)
            new_idx = node_dict[int(i)]
            sub_sparse_adj = sub_adj.to_sparse()
            sub_edge_index_adj = sub_sparse_adj.coalesce().indices()
            print(f"context sub adj: {sub_adj.shape}")
            print(f'control sub features: {sub_feat.shape}')

            print(f"Original prediction for node {i}: {model_predictions[i]}")
            print(f"Prediction label for node {i}: {model_predictions[i].argmax().item()}")
            print(f"New index: {new_idx}")
            with torch.no_grad():
                sub_norm_edge_index = dense_to_sparse(sub_adj)
                print(sub_norm_edge_index)
                sub_model_predictions = model(sub_feat, sub_edge_index_adj)
            print(f"Prediction for subgraph: {sub_model_predictions[new_idx]}")
            print(f"Prediction label for node {i}: {sub_model_predictions[new_idx].argmax().item()}")
            # time.sleep(5)
            CF_ = CFExplainer(model=model,
                                    sub_adj=sub_adj,
                                    sub_feat=sub_feat,
                                    n_hid=64,
                                    dropout=0.01,
                                    sub_labels=sub_labels,
                                    y_pred_orig=y_pred_orig[i],
                                    num_classes = len(labels.unique()),
                                    beta=beta,
                                    device='cpu',
                                    model_type='synthetic')
                
            cf_example = CF_.explain(node_idx=i,
                                        cf_optimizer='SGD', new_idx=new_idx,
                                            lr=0.1,
                                        n_momentum=0.9, num_epochs=500)
            print(f'l EXEMPLE', cf_example)
            if cf_example != []:
                print(f'Counterfactual explanation for node {i}: {cf_example[0][2]}')
                print(f'adjacency matrix: {sub_adj}')
                print(f"counterfactual explanation: {cf_example[0][3]}")
                print(f'number of edges removed: {torch.sum(sub_adj - cf_example[0][2]).item()}')
                # time.sleep(10)
                data.append({"node_idx": i, "X_v": sub_feat, "A_v": sub_adj, "num_edges_removed": torch.sum(sub_adj - cf_example[0][2]).item()})
            else:
                print(f'No counterfactual explanation for node {i}')
                data.append({"node_idx": i, "X_v": sub_feat, "A_v": sub_adj, "num_edges_removed": 0})

        df = pd.DataFrame(data)  # Convert list of dicts to DataFrame
        df.to_pickle(f"data/cf_explanations_beta{beta}.pickle")

        #print average number of edges removed
        print(f"Average number of edges removed: {df['num_edges_removed'].mean()}")
        # Plot histogram showing the distribution of CF examples by the number of edges removed, normalized to show proportions
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.hist(df["num_edges_removed"], bins=20, density=True)  # Use density=True for proportions
        ax.set_title(f"Distribution of Explanation Size for beta {beta}")
        ax.set_xlabel("Number of edges removed")
        ax.set_ylabel("Proportion of CF examples")
        ax.set_xlim(0, 20)
        #between 0 and 20 set ticks every 2
        ax.set_xticks(np.arange(0, 21, 2))
        plt.savefig(f"explanation_size_sparsity_beta{beta}.png")

    #lr
    for lr in lr_values:
        print(f"Learning rate: {lr}")
        model = GCN3Layer_PyG(nfeat=features.shape[1], nhid=64, nout=64,
                     nclass=len(labels.unique()), dropout=0.01)
        state_dict = torch.load("data/model_3layer/GCN3Layer_PyG_citeseer.pt")
        model.load_state_dict(state_dict)
        model.eval()
        data = []
        for i in idx_test[0:10]:
            print(f"Node {i} prediction: {y_pred_orig[i]}")
            sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood_fromRE(int(i), adj, 2, features, labels)
            new_idx = node_dict[int(i)]
            sub_sparse_adj = sub_adj.to_sparse()
            sub_edge_index_adj = sub_sparse_adj.coalesce().indices()
            print(f"context sub adj: {sub_adj.shape}")
            print(f'control sub features: {sub_feat.shape}')

            print(f"Original prediction for node {i}: {model_predictions[i]}")
            print(f"Prediction label for node {i}: {model_predictions[i].argmax().item()}")
            print(f"New index: {new_idx}")
            with torch.no_grad():
                sub_norm_edge_index = dense_to_sparse(sub_adj)
                print(sub_norm_edge_index)
                sub_model_predictions = model(sub_feat, sub_edge_index_adj)
            print(f"Prediction for subgraph: {sub_model_predictions[new_idx]}")
            print(f"Prediction label for node {i}: {sub_model_predictions[new_idx].argmax().item()}")
            # time.sleep(5)
            CF_ = CFExplainer(model=model,
                                    sub_adj=sub_adj,
                                    sub_feat=sub_feat,
                                    n_hid=64,
                                    dropout=0.01,
                                    sub_labels=sub_labels,
                                    y_pred_orig=y_pred_orig[i],
                                    num_classes = len(labels.unique()),
                                    beta=0.5,
                                    device='cpu',
                                    model_type='synthetic')
                
            cf_example = CF_.explain(node_idx=i,
                                        cf_optimizer='SGD', new_idx=new_idx,
                                            lr=lr,
                                        n_momentum=0.9, num_epochs=500)
            print(f'l EXEMPLE', cf_example)
            if cf_example != []:
                print(f'Counterfactual explanation for node {i}: {cf_example[0][2]}')
                print(f'adjacency matrix: {sub_adj}')
                print(f"counterfactual explanation: {cf_example[0][3]}")
                print(f'number of edges removed: {torch.sum(sub_adj - cf_example[0][2]).item()}')
                # time.sleep(10)
                data.append({"node_idx": i, "X_v": sub_feat, "A_v": sub_adj, "num_edges_removed": torch.sum(sub_adj - cf_example[0][2]).item()})
            else:
                print(f'No counterfactual explanation for node {i}')
                data.append({"node_idx": i, "X_v": sub_feat, "A_v": sub_adj, "num_edges_removed": 0})
                
        df = pd.DataFrame(data)  # Convert list of dicts to DataFrame
        df.to_pickle(f"data/cf_explanations_lr{lr}.pickle")

        #print average number of edges removed
        print(f"Average number of edges removed: {df['num_edges_removed'].mean()}")
        # Plot histogram showing the distribution of CF examples by the number of edges removed, normalized to show proportions
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.hist(df["num_edges_removed"], bins=20, density=True)  # Use density=True for proportions
        ax.set_title(f"Distribution of Explanation Size for learning rate {lr}")
        ax.set_xlabel("Number of edges removed")
        ax.set_ylabel("Proportion of CF examples")
        ax.set_xlim(0, 20)
        #between 0 and 20 set ticks every 2
        ax.set_xticks(np.arange(0, 21, 2))
        plt.savefig(f"explanation_size_sparsity_lr{lr}.png")

    #n_momentum
    for n_momentum in n_momentum_values:
        print(f"Momentum: {n_momentum}")
        model = GCN3Layer_PyG(nfeat=features.shape[1], nhid=64, nout=64,
                     nclass=len(labels.unique()), dropout=0.01)
        state_dict = torch.load("data/model_3layer/GCN3Layer_PyG_citeseer.pt")
        model.load_state_dict(state_dict)
        model.eval()
        data = []
        for i in idx_test[0:10]:
            print(f"Node {i} prediction: {y_pred_orig[i]}")
            sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood_fromRE(int(i), adj, 2, features, labels)
            new_idx = node_dict[int(i)]
            sub_sparse_adj = sub_adj.to_sparse()
            sub_edge_index_adj = sub_sparse_adj.coalesce().indices()
            print(f"context sub adj: {sub_adj.shape}")
            print(f'control sub features: {sub_feat.shape}')

            print(f"Original prediction for node {i}: {model_predictions[i]}")
            print(f"Prediction label for node {i}: {model_predictions[i].argmax().item()}")
            print(f"New index: {new_idx}")
            with torch.no_grad():
                sub_norm_edge_index = dense_to_sparse(sub_adj)
                print(sub_norm_edge_index)
                sub_model_predictions = model(sub_feat, sub_edge_index_adj)
            print(f"Prediction for subgraph: {sub_model_predictions[new_idx]}")
            print(f"Prediction label for node {i}: {sub_model_predictions[new_idx].argmax().item()}")
            # time.sleep(5)
            CF_ = CFExplainer(model=model,
                                    sub_adj=sub_adj,
                                    sub_feat=sub_feat,
                                    n_hid=64,
                                    dropout=0.01,
                                    sub_labels=sub_labels,
                                    y_pred_orig=y_pred_orig[i],
                                    num_classes = len(labels.unique()),
                                    beta=0.5,
                                    device='cpu',
                                    model_type='synthetic')
                
            cf_example = CF_.explain(node_idx=i,
                                        cf_optimizer='SGD', new_idx=new_idx,
                                            lr=0.1,
                                        n_momentum=n_momentum, num_epochs=500)
            print(f'l EXEMPLE', cf_example)
            if cf_example != []:
                print(f'Counterfactual explanation for node {i}: {cf_example[0][2]}')
                print(f'adjacency matrix: {sub_adj}')
                print(f"counterfactual explanation: {cf_example[0][3]}")
                print(f'number of edges removed: {torch.sum(sub_adj - cf_example[0][2]).item()}')
                # time.sleep(10)
                data.append({"node_idx": i, "X_v": sub_feat, "A_v": sub_adj, "num_edges_removed": torch.sum(sub_adj - cf_example[0][2]).item()})
            else:
                print(f'No counterfactual explanation for node {i}')
                data.append({"node_idx": i, "X_v": sub_feat, "A_v": sub_adj, "num_edges_removed": 0})

        df = pd.DataFrame(data)  # Convert list of dicts to DataFrame
        df.to_pickle(f"data/cf_explanations_momentum{n_momentum}.pickle")

        #print average number of edges removed
        print(f"Average number of edges removed: {df['num_edges_removed'].mean()}")
        # Plot histogram showing the distribution of CF examples by the number of edges removed, normalized to show proportions
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.hist(df["num_edges_removed"], bins=20, density=True)  # Use density=True for proportions
        ax.set_title(f"Distribution of Explanation Size for momentum {n_momentum}")
        ax.set_xlabel("Number of edges removed")
        ax.set_ylabel("Proportion of CF examples")
        ax.set_xlim(0, 20)
        #between 0 and 20 set ticks every 2
        ax.set_xticks(np.arange(0, 21, 2))
        plt.savefig(f"explanation_size_sparsity_momentum{n_momentum}.png")

    #num_epochs
    for num_epochs in num_epochs_values:
        print(f"Number of epochs: {num_epochs}")
        model = GCN3Layer_PyG(nfeat=features.shape[1], nhid=64, nout=64,
                     nclass=len(labels.unique()), dropout=0.01)
        state_dict = torch.load("data/model_3layer/GCN3Layer_PyG_citeseer.pt")
        model.load_state_dict(state_dict)
        model.eval()
        data = []
        for i in idx_test[0:10]:
            print(f"Node {i} prediction: {y_pred_orig[i]}")
            sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood_fromRE(int(i), adj, 2, features, labels)
            new_idx = node_dict[int(i)]
            sub_sparse_adj = sub_adj.to_sparse()
            sub_edge_index_adj = sub_sparse_adj.coalesce().indices()
            print(f"context sub adj: {sub_adj.shape}")
            print(f'control sub features: {sub_feat.shape}')

            print(f"Original prediction for node {i}: {model_predictions[i]}")
            print(f"Prediction label for node {i}: {model_predictions[i].argmax().item()}")
            print(f"New index: {new_idx}")
            with torch.no_grad():
                sub_norm_edge_index = dense_to_sparse(sub_adj)
                print(sub_norm_edge_index)
                sub_model_predictions = model(sub_feat, sub_edge_index_adj)
            print(f"Prediction for subgraph: {sub_model_predictions[new_idx]}")
            print(f"Prediction label for node {i}: {sub_model_predictions[new_idx].argmax().item()}")
            # time.sleep(5)
            CF_ = CFExplainer(model=model,
                                    sub_adj=sub_adj,
                                    sub_feat=sub_feat,
                                    n_hid=64,
                                    dropout=0.01,
                                    sub_labels=sub_labels,
                                    y_pred_orig=y_pred_orig[i],
                                    num_classes = len(labels.unique()),
                                    beta=0.5,
                                    device='cpu',
                                    model_type='synthetic')
                
            cf_example = CF_.explain(node_idx=i,
                                        cf_optimizer='SGD', new_idx=new_idx,
                                            lr=0.1,
                                        n_momentum=0.9, num_epochs=num_epochs)
            print(f'l EXEMPLE', cf_example)
            if cf_example != []:
                print(f'Counterfactual explanation for node {i}: {cf_example[0][2]}')
                print(f'adjacency matrix: {sub_adj}')
                print(f"counterfactual explanation: {cf_example[0][3]}")
                print(f'number of edges removed: {torch.sum(sub_adj - cf_example[0][2]).item()}')
                # time.sleep(10)
                data.append({"node_idx": i, "X_v": sub_feat, "A_v": sub_adj, "num_edges_removed": torch.sum(sub_adj - cf_example[0][2]).item()})
            else:
                print(f'No counterfactual explanation for node {i}')
                data.append({"node_idx": i, "X_v": sub_feat, "A_v": sub_adj, "num_edges_removed": 0})

        df = pd.DataFrame(data)  # Convert list of dicts to DataFrame
        df.to_pickle(f"data/cf_explanations_epochs{num_epochs}.pickle")

        #print average number of edges removed
        print(f"Average number of edges removed: {df['num_edges_removed'].mean()}")
        # Plot histogram showing the distribution of CF examples by the number of edges removed, normalized to show proportions
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.hist(df["num_edges_removed"], bins=20, density=True)  # Use density=True for proportions
        ax.set_title(f"Distribution of Explanation Size for number of epochs {num_epochs}")
        ax.set_xlabel("Number of edges removed")
        ax.set_ylabel("Proportion of CF examples")
        ax.set_xlim(0, 20)
        #between 0 and 20 set ticks every 2
        ax.set_xticks(np.arange(0, 21, 2))
        plt.savefig(f"explanation_size_sparsity_epochs{num_epochs}.png")
 


if __name__ == "__main__":
    main()
