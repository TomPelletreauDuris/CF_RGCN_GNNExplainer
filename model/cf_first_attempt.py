import torch
import numpy as np
from torch.autograd import Variable

# Import necessary modules from other scripts
from Data_loading import load_cora_dataset
from Re_GNNExplainer.model.GCN import GCN, normalize_adjacency_matrix
from Preprocessing import create_adjacency_matrix
from utils import select_node_by_confidence, select_node_by_uncertainty, load_node_specific_data, get_2_hop_neighborhood, extract_subgraph
from Model import get_the_model, compute_loss, compute_accuracy


def sigmoid(X):
    return 1 / (1 + torch.exp(-X))    

def threshold(P_hat):
    P = (P_hat >= 0.5).float()
    return P

def compute_perturbed_adjacency(A_v, P):
    return P * A_v

def compute_loss_CF(f_v, f_v_bar, y_true, beta, distance):
    print(f"f_v: {f_v}")
    print(f"f_v_bar: {f_v_bar}")
    # Convert logits/probabilities to predicted classes
    predicted_class_f_v = f_v.argmax(dim=1)
    predicted_class_f_v_bar = f_v_bar.argmax(dim=1)
    print(f"predicted_class_f_v: {predicted_class_f_v}")
    print(f"predicted_class_f_v_bar: {predicted_class_f_v_bar}")
    
    # Check if the predicted classes are different
    if torch.equal(predicted_class_f_v, predicted_class_f_v_bar):
        loss_pred = -torch.nn.functional.nll_loss(f_v_bar, y_true)
    else:
        loss_pred = 0
    loss_dist = beta * distance
    return loss_pred + loss_dist

def compute_distance(A_v, A_v_bar):
    # return torch.sum(torch.abs(A_v - A_v_bar)) / 2  # Divide by 2 for undirected graphs
    #L2 norm of the difference between the adjacency matrices
    return torch.norm(A_v - A_v_bar, p=2)



def main():
    """
    Main function to generate counterfactual explanations for a node in the Cora dataset.
    """
        
    #use get_the_model to get the model and the data
    model, num_nodes, num_features, num_labels, X, A, A_hat, y_true, y_pred, train_idx, test_idx, W1, W2 = get_the_model()
    print(f'model trained and data loaded')
    model.eval()
    model_predictions = model(X, A_hat)
    #is y_pred the same as model_predictions?
    print(f'y_pred: {y_pred}')
    print(f'model_predictions: {model_predictions}')
    test_loss = compute_loss(model_predictions[test_idx], y_true[test_idx])
    test_acc = compute_accuracy(model_predictions.argmax(dim=1)[test_idx], y_true[test_idx])
    print(f'CHECK Test Loss: {test_loss.item():0.4f}, Test Acc: {test_acc:0.4f}')
    true_labels = y_true
    full_adjacency_matrix = A

    # Select the node with the highest confidence
    node_id = select_node_by_confidence(model_predictions, true_labels, full_adjacency_matrix,  threshold=0.9)
    print(f"Selected node ID with confidence: {node_id}")
    # A_v, X_v, y_true = load_node_specific_data(node_id, full_adjacency_matrix, X)

    neighborhood = get_2_hop_neighborhood(node_id[0], full_adjacency_matrix)
    A_v, X_v = extract_subgraph(neighborhood, full_adjacency_matrix, X)
    print(f"Number of nodes in the neighborhood: {A_v.shape[0]}")
    print(f"Number of edges in the neighborhood: {int(torch.sum(A_v) / 2)}")
    print(f"Number of features per node: {X_v.shape[1]}")

    y_true_id = true_labels[node_id[0]]
    print(f"True label of the node: {y_true_id}")
    print(f"Predicted label of the node: {model_predictions[node_id[0]].argmax(dim=0)}")

    # Compute the prediction for the node
    f_v = model(X_v, A_v)
    print(f"Prediction for the node: {f_v.argmax(dim=1)}")

    # Parameters as per your specification
    k = 500
    beta = 0.5
    alpha = 0.1

    # Counterfactual explanation generation loop
    P_hat = Variable(torch.randn(A_v.shape), requires_grad=True)  # Initialize P_hat as learnable
    optimizer = torch.optim.Adam([P_hat], lr=alpha)

    for iteration in range(k):
        # Zero out gradients
        optimizer.zero_grad()

        # Compute binary perturbation matrix P
        P = threshold(sigmoid(P_hat))
        A_v_bar = compute_perturbed_adjacency(A_v, P)
        print(f"Normal adjacency matrix: {A_v}")
        print(f"Perturbed adjacency matrix: {A_v_bar}")

        # Generate prediction for perturbed adjacency matrix
        f_v = model(X_v, A_v)
        f_v_bar = model(X_v, A_v_bar)

        # Compute distance between the original and perturbed adjacency matrices
        distance = compute_distance(A_v, A_v_bar)
        print(f"Distance between adjacency matrices: {distance}")

        
        # Ensure neighborhood is a list or convert it to a list
        if not isinstance(neighborhood, list):
            neighborhood = list(neighborhood)
        y_true_sub = true_labels[neighborhood]
        #Ensure that The length of true_labels[neighborhood] matches the number of nodes for which you have predictions in f_v.
        print(f"y_true_sub: {y_true_sub}")
        print(f"y_true_sub.shape: {y_true_sub.shape}")

        # Compute loss
        loss = compute_loss_CF(f_v, f_v_bar, y_true_sub, beta, distance)
        print(f"Loss: {loss}")

        # Backpropagation
        if loss.requires_grad:  # Check if loss requires gradient
            loss.backward()  # Compute gradients
            optimizer.step()  # Update P_hat based on gradients
        else:
            print("Loss does not have grad_fn, iteration:", iteration)

        #update and print P_hat
        print(f"P_hat: {P_hat}")
    
        # Check if the prediction has changed; if so, this is a candidate counterfactual
        if not torch.equal(f_v.argmax(dim=1), f_v_bar.argmax(dim=1)):
            print(f"Found counterfactual at iteration {iteration}")
            # break  # Stop if we found a counterfactual; remove this if you want to find the minimal one

    # After finding the counterfactual, we might save or print it
    # Save or process the perturbed adjacency matrix A_v_bar as the counterfactual explanation
    torch.save(A_v_bar, '../data/counterfactual.pt')

if __name__ == "__main__":
    main()
