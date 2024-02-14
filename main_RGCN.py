import torch
import numpy as np
from torch.autograd import Variable
import copy
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import time
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F
import sys
# sys.path.append('model')

# Import necessary modules from other scripts

from model.GCN import RGCN
from model.utils import  get_neighbourhood_fromRGCN
from model.cf_RGCN import  R_CFExplainer


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

    #Load the AIFB dataset

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

    #true_labels_train different occurences of the same label
    print(f"True labels train: {true_labels_train.unique()}")
    

    #define GCN model
    model = RGCN(num_entities=num_nodes, num_relations=num_relations, num_classes=num_classes)
    model.load_state_dict(torch.load('data/R-model_2layer/R-model.pt'))
    model.eval()

    print(A.shape)
    print(X.shape)
    print(logits.shape)
    print(f'Number of nodes: {num_nodes}, Number of relations: {num_relations}, Number of classes: {num_classes}')
    print(f'Edge index: {edge_index.shape}, Edge type: {edge_type.shape}')
    print(f'Number of training nodes: {len(train_idx)}')
    print(f'Number of test nodes: {len(test_idx)}')
    print(f"control edge_index: {edge_index.unique()}")
    print(f'control edge_type: {edge_type.unique()}')
    print("___________________")

    with torch.no_grad():
        model_predictions = model.forward(edge_index, edge_type)
    loss_orig = F.cross_entropy(logits[train_idx], true_labels_train)
    loss = F.cross_entropy(model_predictions[train_idx], true_labels_train)
    print(f'Original Loss: {loss_orig.item():0.4f}, Loss: {loss.item():0.4f}')

    train_acc = torch.sum(logits[train_idx].argmax(dim=1) == true_labels_train)
    train_acc = train_acc.item() / len(train_idx)
    test_acc = torch.sum(logits[test_idx].argmax(dim=1) == true_labels_test)
    test_acc = test_acc / len(test_idx)
    print("logits" +
        "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
            train_acc, loss.item()) +
        "Test Accuracy: {:.4f} ".format(test_acc))
    train_acc = torch.sum(model_predictions[train_idx].argmax(dim=1) == true_labels_train)
    train_acc = train_acc.item() / len(train_idx)
    test_acc = torch.sum(model_predictions[test_idx].argmax(dim=1) == true_labels_test)
    test_acc = test_acc / len(test_idx)
    print("model_predictions" +
        "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
            train_acc, loss.item()) +
        "Test Accuracy: {:.4f} ".format(test_acc))
    
    y_pred_orig = torch.argmax(model_predictions, dim=1)
    print("y_pred length: ", len(y_pred_orig))
    print("y_pred counts: ", y_pred_orig.unique(return_counts=True))

    # exit()
    test_idx = test_idx[0:10]
    text_idx_same = []

    for idx in test_idx:
        print(f"Node {idx} prediction: {y_pred_orig[idx]}")
        sub_edge_index, sub_edge_type, sub_feat, node_dict = get_neighbourhood_fromRGCN(int(idx), edge_index, edge_type, 4, X)
        new_idx = node_dict[int(idx)]
        with torch.no_grad():
            sub_model_predictions = model.forward(sub_edge_index, sub_edge_type)
        print(f"Original prediction for node {idx}: {model_predictions[idx]}")
        print(f"Prediction for subgraph: {sub_model_predictions[new_idx]}")
        if model_predictions[idx].argmax().item() == sub_model_predictions[new_idx].argmax().item():
            text_idx_same.append(idx)

    data = []  # Initialize an empty list to collect data

    for idx in text_idx_same:
        print(f"Node {idx} prediction: {y_pred_orig[idx]}")
        sub_edge_index, sub_edge_type, sub_feat, node_dict = get_neighbourhood_fromRGCN(int(idx), edge_index, edge_type, 4, X)
        new_idx = node_dict[int(idx)]
        sub_adj = to_dense_adj(sub_edge_index)[0]
        
        """ 
        # edge_index and edge_type are defined and correspond to the entire graph
        for i, (source, target) in enumerate(sub_edge_index.t().tolist()):
            original_source = [k for k, v in node_dict.items() if v == source][0]
            original_target = [k for k, v in node_dict.items() if v == target][0]
            # Find the edge in the original edge_index and assign its type to sub_edge_type
            edge_pos = (edge_index[0] == original_source) & (edge_index[1] == original_target)
            if edge_pos.any():
                sub_edge_type[i] = edge_type[edge_pos.nonzero(as_tuple=True)[0][0]] """
        print(f"control sub_edge_index: {sub_edge_index.shape}")
        print(f'control sub_edge_type: {sub_edge_type.shape}')
        print(f"control sub_edge_index: {sub_edge_index.unique()}")
        print(f'control sub_edge_type: {sub_edge_type.unique()}')

        sub_num_nodes = sub_feat.size(0)
        sub_num_relations = sub_edge_type.unique().size(0)

        print(f"Original prediction for node {idx}: {model_predictions[idx]}")
        print(f"Prediction label for node {idx}: {model_predictions[idx].argmax().item()}")
        print(f"New index: {new_idx}")
        with torch.no_grad():
            sub_model_predictions = model.forward(sub_edge_index, sub_edge_type)
        print(f"Prediction for subgraph: {sub_model_predictions[new_idx]}")
        print(f"Prediction label for node {idx}: {sub_model_predictions[new_idx].argmax().item()}")

        CF_ = R_CFExplainer(model=model, 
                            num_entities = num_nodes,
                            num_relations = num_relations,
                            num_classes = num_classes,
                            sub_edge_index = sub_edge_index,
                            sub_edge_type = sub_edge_type,
                            num_nodes= sub_num_nodes,
                            y_pred_orig = y_pred_orig[idx],
                            dropout = 0.01,
                            beta = 0.5,
                            model_type = 'RGCN')
            
        cf_example = CF_.explain(node_idx=idx,
                                    cf_optimizer='SGD', new_idx=new_idx,
                                        lr=0.5,
                                    n_momentum=0.999, num_epochs=500)
        print(f'l EXEMPLE', cf_example)
        if cf_example != []:
            print(f'Counterfactual explanation for node {idx}: {cf_example[0][2]}')
            print(f'adjacency matrix: {sub_adj}')
            print(f"counterfactual explanation: {cf_example[0][3]}")
            print(f'number of edges removed: {torch.sum(sub_adj - cf_example[0][2]).item()}')
            time.sleep(10)
    
            data.append({"node_idx": idx, "X_v": sub_feat, "A_v": sub_adj, "num_edges_removed": torch.sum(sub_adj - cf_example[0][2]).item()/2})
        else:
            print(f'No counterfactual explanation for node {idx}')
            data.append({"node_idx": idx, "X_v": sub_feat, "A_v": sub_adj, "num_edges_removed": 0})
    
    df = pd.DataFrame(data)  # Convert list of dicts to DataFrame
    df.to_pickle("data/cf_explanations_test_full_final.pickle")
    
    #print average number of edges removed
    print(f"Average number of edges removed: {df['num_edges_removed'].mean()}")

    #plot Histograms showing the proportion of CF examples for the number of edges removed 
    
    # Plot histogram showing the distribution of CF examples by the number of edges removed, normalized to show proportions
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(df["num_edges_removed"], bins=20, density=True)  # Use density=True for proportions
    ax.set_title("Distribution of Explanation Size")
    ax.set_xlabel("Number of edges removed")
    ax.set_ylabel("Proportion of CF examples")
    ax.set_xlim(0, 20)
    #between 0 and 20 set ticks every 2
    ax.set_xticks(np.arange(0, 21, 2))
    plt.savefig("explanation_size_sparsity_fullCF_SGD_D005_lr05_b055_mm00.png")
    plt.show()


if __name__ == "__main__":
    main()
