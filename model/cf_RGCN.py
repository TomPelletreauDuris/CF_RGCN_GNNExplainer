import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn.parameter import Parameter
from torch.optim import Adam
import time
import torch.optim as optim
import math
from model.GCN import GraphConvolutionLayer, RGCN
from torchviz import make_dot
from torch.distributions import RelaxedOneHotCategorical

def edge_index_to_dense_adj(edge_index, num_nodes):
    # edge_index: [2, num_edges]
    # num_nodes: int, the number of nodes in the graph
    
    # Create a tensor of zeros with shape [num_nodes, num_nodes]
    dense_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    
    # Use the edge_index to fill in the adjacency matrix
    # This operation broadcasts the value '1' into the dense_adj at the positions specified by edge_index
    dense_adj[edge_index[0], edge_index[1]] = 1.0
    
    return dense_adj

def create_vec_from_symm_matrix(matrix, P_vec_size):
    idx = torch.tril_indices(row=matrix.size(0), col=matrix.size(1), offset=0)
    vector = matrix[idx[0], idx[1]]
    return vector

def create_symm_matrix_from_vec(vector, n_rows):
    # Initialize a matrix to fill in; use requires_grad if necessary
    matrix = torch.zeros((n_rows, n_rows), dtype=vector.dtype, device=vector.device)
    # Calculate the indices for the lower triangular part
    idx = torch.tril_indices(row=n_rows, col=n_rows, offset=0)
    # Assign the vector values to the lower triangular part
    matrix[idx[0], idx[1]] = vector
    # Make the matrix symmetric
    symm_matrix = matrix + matrix.transpose(0, 1) - torch.diag(torch.diag(matrix))
    return symm_matrix

def get_degree_matrix(adj):
    return torch.diag(sum(adj))
       
#similar to GraphConvolution
class GraphConvolutionPerturb(nn.Module):
    """
    Similar to GraphConvolution
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionPerturb, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias is not None:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)


    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
# Our perturbation model where we add a perturbation to the adjacency matrix
class GCNPerturb(nn.Module):
    """
    2-layer GCN perturbed by P_hat
    
    """
    def __init__(self, nfeat, nhid, nclass, adj, dropout, beta, edge_additions=False):
        super(GCNPerturb, self).__init__()
        self.adj = adj
        self.beta = beta
        self.num_nodes = self.adj.shape[0]
        self.edge_additions = edge_additions
        self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2) + self.num_nodes

        self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))

        self.reset_parameters()
        print(f"P VECTOR !",self.P_vec)

        self.conv1 = GraphConvolutionPerturb(nfeat, nhid, bias=False)
        self.conv2 = GraphConvolutionLayer(nhid, nclass, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def reset_parameters(self, eps=1e-4):
        # Reset parameters logic
        with torch.no_grad():
            torch.sub(self.P_vec, eps)


    def forward(self, given_sub_feat, adj):
        print('forward ')
        self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)
        print(f"P_hat_symm : {self.P_hat_symm}")
        A_tilde = F.sigmoid(self.P_hat_symm) * self.adj + torch.eye(self.num_nodes)
        D_tilde = get_degree_matrix(A_tilde)
        D_tilde_exp = D_tilde ** (-0.5)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        x = self.conv1(given_sub_feat, A_tilde)
        x = self.relu(x)
        x = F.dropout(x) #HERE we put the dropout to comment because we don't want to put the model in train mode but still have a coherent output , training=False
        x = self.conv2(x, A_tilde)
        return self.log_softmax(x)

    def forward_prediction(self, given_sub_feat):
        print('forward_prediction ')
        self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)
        self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()
        print(f"P : {self.P}")
        print(f"P_hat_symm : {self.P_hat_symm}")
        print(f"self.adj : {self.adj}")
        A_tilde = self.P * self.adj + torch.eye(self.num_nodes)
        print(f"A_tilde : {A_tilde}")

        D_tilde = get_degree_matrix(A_tilde)
        D_tilde_exp = D_tilde ** (-0.5)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        x = self.conv1(given_sub_feat, A_tilde)
        # print(f"First conv layer in PERTURB : {x}")
        x = self.relu(x)
        # print(f"Relu in PERTURB : {x}")
        x = F.dropout(x) #, training=False
        print(f"Dropout in PERTURB : {x}")
        x = self.conv2(x, A_tilde)
        return self.log_softmax(x), self.P

    def loss(self, output, y_pred_orig, y_pred_new_actual):
        pred_same = (y_pred_new_actual == y_pred_orig).float()
        print(pred_same)

        # Need dim >=2 for F.nll_loss to work
        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)

        cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        print(output)
        print(y_pred_orig)
        t1 = [-8.0272, -7.9911, -3.1536, -1.3117, -0.8077, -1.4214]
        t1 = torch.tensor(t1)
        t1 = t1.unsqueeze(0)
        t2 = [4]
        t2 = torch.tensor(t2)
        # t2 = t2.unsqueeze(0)
        loss_pred = - F.nll_loss(t1, t2)
        print(f"loss_pred : {loss_pred}")
        loss_pred = - F.nll_loss(output, y_pred_orig)
        print(f"check the nature of adj before loss : {self.adj}")
        loss_graph_dist = sum(sum(abs(cf_adj - self.adj))) #/ 2      # Number of edges changed (symmetrical)

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss_total, loss_pred, loss_graph_dist, cf_adj


class GCN2LayerPerturb(nn.Module):
    """
    2-layer GCN used in GNN Explainer
    """
    def __init__(self, nfeat, nhid, nout, nclass, adj, dropout, beta, edge_additions=False):
        super(GCN2LayerPerturb, self).__init__()
        self.adj = adj
        self.nclass = nclass
        self.beta = beta
        self.num_nodes = self.adj.shape[0]
        self.edge_additions = edge_additions      # are edge additions included in perturbed matrix

        # P_hat needs to be symmetric ==> learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
        self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2)  + self.num_nodes

        if self.edge_additions:
            self.P_vec = Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size)))
        else:
            self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))

        self.reset_parameters()

        self.gc1 = GraphConvolutionPerturb(nfeat, nhid)
        self.gc2 = GraphConvolutionLayer(nhid, nhid)
        self.lin = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def reset_parameters(self, eps=10**-4):
        # Think more about how to initialize this
        with torch.no_grad():
            if self.edge_additions:
                adj_vec = create_vec_from_symm_matrix(self.adj, self.P_vec_size).numpy()
                for i in range(len(adj_vec)):
                    if i < 1:
                        adj_vec[i] = adj_vec[i] - eps
                    else:
                        adj_vec[i] = adj_vec[i] + eps
                torch.add(self.P_vec, torch.FloatTensor(adj_vec))       #self.P_vec is all 0s
            else:
                torch.sub(self.P_vec, eps)

    def forward(self, x, sub_adj):
        self.sub_adj = sub_adj
        # Same as normalize_adj in utils.py except includes P_hat in A_tilde
        self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)      # Ensure symmetry

        A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True

        if self.edge_additions:         # Learn new adj matrix directly
            A_tilde = F.sigmoid(self.P_hat_symm) + torch.eye(self.num_nodes)  # Use sigmoid to bound P_hat in [0,1]
        else:       # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
            A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj + torch.eye(self.num_nodes)       # Use sigmoid to bound P_hat in [0,1]

        D_tilde = get_degree_matrix(A_tilde).detach()       # Don't need gradient of this
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        x1 = F.relu(self.gc1(x, norm_adj))
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, norm_adj))
        #x2 = F.dropout(x2, self.dropout, training=self.training)
        x = self.lin(x2)
        #x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


    def forward_prediction(self, x):
        # Same as forward but uses P instead of P_hat ==> non-differentiable
        # but needed for actual predictions

        self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()      # threshold P_hat

        if self.edge_additions:
            A_tilde = self.P + torch.eye(self.num_nodes)
        else:
            A_tilde = self.P * self.adj + torch.eye(self.num_nodes)

        D_tilde = get_degree_matrix(A_tilde)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        x1 = F.relu(self.gc1(x, norm_adj))
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, norm_adj))
        #x2 = F.dropout(x2, self.dropout, training=self.training)
        x = self.lin(x2)
        #x = F.dropout(x, self.dropout, training=self.training)

        return F.log_softmax(x, dim=1), self.P


    def  loss(self, output, y_pred_orig, y_pred_new_actual):
        pred_same = (y_pred_new_actual == y_pred_orig).float()
        print(f'pred same', pred_same)

        # Need dim >=2 for F.nll_loss to work
        print('-----------------------------')
        print(f'output : {output}')
        print(f'y_pred_orig : {y_pred_orig}')
        # output = output.detach()
        output = output.unsqueeze(0)
        # output = output.squeeze()
        y_pred_orig = y_pred_orig.unsqueeze(0)
        y_pred_orig = y_pred_orig.view(-1)
        # y_pred_orig = y_pred_orig.unsqueeze(0)
        print(f'output : {output}')
        print(f'y_pred_orig : {y_pred_orig}')

        if self.edge_additions:
            cf_adj = self.P
        else:
            cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = - F.nll_loss(output, y_pred_orig).float()
        loss_graph_dist = sum(sum(abs(cf_adj - self.adj))) / 2      # Number of edges changed (symmetrical)

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss_total, loss_pred, loss_graph_dist, cf_adj

class GCNSyntheticPerturb_synthetic(nn.Module):
    """
    3-layer GCN used in GNN Explainer synthetic tasks
    """
    def __init__(self, nfeat, nhid, nout, nclass, adj, dropout, beta, edge_additions=False):
        super(GCNSyntheticPerturb_synthetic, self).__init__()
        self.adj = adj
        self.nclass = nclass
        self.beta = beta
        self.num_nodes = self.adj.shape[0]
        self.edge_additions = edge_additions      # are edge additions included in perturbed matrix

        # P_hat needs to be symmetric ==> learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
        self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2)  + self.num_nodes

        if self.edge_additions:
            self.P_vec = Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size)))
        else:
            self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))

        self.reset_parameters()
        print(f"P VECTOR !",self.P_vec)

        self.gc1 = GraphConvolutionPerturb(nfeat, nhid)
        self.gc2 = GraphConvolutionPerturb(nhid, nhid)
        self.gc3 = GraphConvolutionPerturb(nhid, nhid)
        self.lin = nn.Linear(60, nclass)
        self.dropout = dropout

    def reset_parameters(self, eps=10**-2):
        # Think more about how to initialize this
        with torch.no_grad():
            if self.edge_additions:
                adj_vec = create_vec_from_symm_matrix(self.adj, self.P_vec_size).numpy()
                for i in range(len(adj_vec)):
                    if i < 1:
                        adj_vec[i] = adj_vec[i] - eps
                    else:
                        adj_vec[i] = adj_vec[i] + eps
                torch.add(self.P_vec, torch.FloatTensor(adj_vec))       #self.P_vec is all 0s
            else:
                torch.sub(self.P_vec, eps)




    def forward(self, x, sub_adj):
        self.sub_adj = sub_adj
        # Same as normalize_adj in utils.py except includes P_hat in A_tilde
        self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)      # Ensure symmetry
        # print(f"check p_hat_symm : {self.P_hat_symm}")
        A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True

        # print(f"check if sub_adj is the same : {sub_adj}")

        A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj + torch.eye(self.num_nodes)       # Use sigmoid to bound P_hat in [0,1]

        D_tilde = get_degree_matrix(A_tilde).detach()       # Don't need gradient of this
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        x1 = F.relu(self.gc1(x, norm_adj))
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, norm_adj))
        #x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, norm_adj)
        x = self.lin(x3)
        # print(f"check if x is the same, first line : {x[25]}")
        x = F.dropout(x, self.dropout, training=self.training)
        # print(f"check if x is the same, second line : {x[25]}")
        return F.log_softmax(x, dim=1)


    def forward_prediction(self, x):
        
        # Same as forward but uses P instead of P_hat ==> non-differentiable
        # but needed for actual predictions
        # self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)      # Ensure symmetry

        self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()      # threshold P_hat
        # print(f"check if P is the same : {self.P}")

        # print(f"check if sub_adj is the same : {self.sub_adj}")
        A_tilde = self.P * self.sub_adj + torch.eye(self.num_nodes)

        D_tilde = get_degree_matrix(A_tilde).detach()     
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        x1 = F.relu(self.gc1(x, norm_adj))
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, norm_adj))
        #x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, norm_adj)
        x = self.lin(x3)
        # print(f"check if x is the same : {x[25]}")
        x = F.dropout(x, self.dropout, training=self.training)
        # print(f"check if x is the same : {x[25]}")
        return F.log_softmax(x, dim=1), self.P


    def  loss(self, output, y_pred_orig, y_pred_new_actual):
        pred_same = (y_pred_new_actual == y_pred_orig).float()
        print(pred_same)
        if pred_same == 0:
            time.sleep(0)

        # Need dim >=2 for F.nll_loss to work
        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)

        cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient
        
        print(f"CF ADJ TO CHECK : {cf_adj}")
        
        # Want negative in front to maximize loss instead of minimizing it to find CFs
        print(output)
        print(y_pred_orig)

        loss_pred = - F.nll_loss(output, y_pred_orig)
        print(self.P)
        print(self.P_hat_symm)
        # print(f"check the nature of adj before loss : {self.adj}")
        # print(f"check the nature of cf_adj before loss : {cf_adj}")
        #if adj is zeros
        if torch.sum(self.adj) == 0:
            time.sleep(10)
        #if adj has a line of zeros
        if torch.sum(self.adj, dim=1).any() == 0:
            time.sleep(10)

        #if max p_vex is > 1.1 let's put a penalty
        if torch.max(self.P_vec) > 1.01:
            regularization_term = 0.1 * (torch.max(self.P_vec) - 1)
            # time.sleep(10)
        else :
            regularization_term = 0


        loss_graph_dist = sum(sum(abs(cf_adj - self.adj))) #/ 2      # Number of edges changed (symmetrical)        
        # Zero-out loss_pred with pred_same if prediction flips
        print(f"pred_same * loss_pred : {pred_same * loss_pred}")
        print(f"self.beta * loss_graph_dist : {self.beta * loss_graph_dist}")
        loss_total = (-pred_same) * loss_pred + self.beta * loss_graph_dist
        loss_total = loss_total + regularization_term
                
        return loss_total, loss_pred, loss_graph_dist, cf_adj

#replication of the GCN model with the perturbation
class GCNSyntheticPerturb(nn.Module):
    """
    3-layer GCN used in GNN Explainer synthetic tasks
    """
    def __init__(self, nfeat, nhid, nout, nclass, adj, dropout, beta, edge_additions=False):
        super(GCNSyntheticPerturb, self).__init__()
        self.adj = adj
        self.nclass = nclass
        self.beta = beta
        self.num_nodes = self.adj.shape[0]
        self.edge_additions = edge_additions      # are edge additions included in perturbed matrix

        # P_hat needs to be symmetric ==> learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
        self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2)  + self.num_nodes

        if self.edge_additions:
            self.P_vec = Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size)))
        else:
            self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))

        self.reset_parameters()
        print(f"P VECTOR !",self.P_vec)

        self.gc1 = GraphConvolutionPerturb(nfeat, nhid)
        self.gc2 = GraphConvolutionPerturb(nhid, nhid)
        self.gc3 = GraphConvolutionLayer(nhid, nhid)
        self.lin = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def reset_parameters(self, eps=10**-2):
        # Think more about how to initialize this
        with torch.no_grad():
            if self.edge_additions:
                adj_vec = create_vec_from_symm_matrix(self.adj, self.P_vec_size).numpy()
                for i in range(len(adj_vec)):
                    if i < 1:
                        adj_vec[i] = adj_vec[i] - eps
                    else:
                        adj_vec[i] = adj_vec[i] + eps
                torch.add(self.P_vec, torch.FloatTensor(adj_vec))       #self.P_vec is all 0s
            else:
                torch.sub(self.P_vec, eps)




    def forward(self, x, sub_adj):
        self.sub_adj = sub_adj
        # Same as normalize_adj in utils.py except includes P_hat in A_tilde
        self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)      # Ensure symmetry
        # print(f"check p_hat_symm : {self.P_hat_symm}")
        A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True

        # print(f"check if sub_adj is the same : {sub_adj}")

        A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj + torch.eye(self.num_nodes)       # Use sigmoid to bound P_hat in [0,1]

        D_tilde = get_degree_matrix(A_tilde).detach()       # Don't need gradient of this
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        x1 = F.relu(self.gc1(x, norm_adj))
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, norm_adj))
        #x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, norm_adj)
        x = self.lin(x3)
        # print(f"check if x is the same, first line : {x[25]}")
        x = F.dropout(x, self.dropout, training=self.training)
        # print(f"check if x is the same, second line : {x[25]}")
        return F.log_softmax(x, dim=1)


    def forward_prediction(self, x):
        
        # Same as forward but uses P instead of P_hat ==> non-differentiable
        # but needed for actual predictions
        # self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)      # Ensure symmetry

        self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()      # threshold P_hat
        # print(f"check if P is the same : {self.P}")

        # print(f"check if sub_adj is the same : {self.sub_adj}")
        A_tilde = self.P * self.sub_adj + torch.eye(self.num_nodes)

        D_tilde = get_degree_matrix(A_tilde).detach()     
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        x1 = F.relu(self.gc1(x, norm_adj))
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, norm_adj))
        #x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, norm_adj)
        x = self.lin(x3)
        # print(f"check if x is the same : {x[25]}")
        x = F.dropout(x, self.dropout, training=self.training)
        # print(f"check if x is the same : {x[25]}")
        return F.log_softmax(x, dim=1), self.P


    def  loss(self, output, y_pred_orig, y_pred_new_actual):
        pred_same = (y_pred_new_actual == y_pred_orig).float()
        print(pred_same)
        if pred_same == 0:
            time.sleep(0)

        # Need dim >=2 for F.nll_loss to work
        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)

        cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient
        
        print(f"CF ADJ TO CHECK : {cf_adj}")
        
        # Want negative in front to maximize loss instead of minimizing it to find CFs
        print(output)
        print(y_pred_orig)

        loss_pred = - F.nll_loss(output, y_pred_orig)
        print(self.P)
        print(self.P_hat_symm)
        # print(f"check the nature of adj before loss : {self.adj}")
        # print(f"check the nature of cf_adj before loss : {cf_adj}")
        #if adj is zeros
        if torch.sum(self.adj) == 0:
            time.sleep(10)
        #if adj has a line of zeros
        if torch.sum(self.adj, dim=1).any() == 0:
            time.sleep(10)

        #if max p_vex is > 1.1 let's put a penalty
        if torch.max(self.P_vec) > 1.01:
            regularization_term = 0.1 * (torch.max(self.P_vec) - 1)
            # time.sleep(10)
        else :
            regularization_term = 0


        loss_graph_dist = sum(sum(abs(cf_adj - self.adj))) #/ 2      # Number of edges changed (symmetrical)        
        # Zero-out loss_pred with pred_same if prediction flips
        print(f"pred_same * loss_pred : {pred_same * loss_pred}")
        print(f"self.beta * loss_graph_dist : {self.beta * loss_graph_dist}")
        loss_total = (-pred_same) * loss_pred + self.beta * loss_graph_dist
        loss_total = loss_total + regularization_term
                
        return loss_total, loss_pred, loss_graph_dist, cf_adj

""" class RGCNperturb_2(nn.Module):
    def __init__(self, nfeat, nhid, nout, nclass, adj, dropout, beta, edge_additions=False):
        super(RGCNperturb_2, self).__init__()
        self.adj = adj
        self.beta = beta
        self.num_nodes = self.adj.shape[0]
        self.edge_additions = edge_additions
        self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2) + self.num_nodes

        self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))

        self.reset_parameters()
        print(f"P VECTOR !",self.P_vec)

        self.conv1 = GraphConvolutionPerturb(nfeat, nhid, bias=False)
        self.conv2 = GraphConvolutionLayer(nhid, nclass, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def reset_parameters(self, eps=1e-4):
        # Reset parameters logic
        with torch.no_grad():
            torch.sub(self.P_vec, eps)


    def forward(self, given_sub_feat, adj):
        print('forward ')
        self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)
        print(f"P_hat_symm : {self.P_hat_symm}")
        A_tilde = F.sigmoid(self.P_hat_symm) * self.adj + torch.eye(self.num_nodes)
        D_tilde = get_degree_matrix(A_tilde)
        D_tilde_exp = D_tilde ** (-0.5)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        x = self.conv1(given_sub_feat, A_tilde)
        x = self.relu(x)
        x = F.dropout(x) #HERE we put the dropout to comment because we don't want to put the model in train mode but still have a coherent output , training=False
        x = self.conv2(x, A_tilde)
        return self.log_softmax(x)

    def forward_prediction(self, given_sub_feat):
        print('forward_prediction ')
        self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)
        self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()
        print(f"P : {self.P}")
        print(f"P_hat_symm : {self.P_hat_symm}")
        print(f"self.adj : {self.adj}")
        A_tilde = self.P * self.adj + torch.eye(self.num_nodes)
        print(f"A_tilde : {A_tilde}")

        D_tilde = get_degree_matrix(A_tilde)
        D_tilde_exp = D_tilde ** (-0.5)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        x = self.conv1(given_sub_feat, A_tilde)
        # print(f"First conv layer in PERTURB : {x}")
        x = self.relu(x)
        # print(f"Relu in PERTURB : {x}")
        x = F.dropout(x) #, training=False
        print(f"Dropout in PERTURB : {x}")
        x = self.conv2(x, A_tilde)
        return self.log_softmax(x), self.P """
    


# class RGCNPerturb(nn.Module):
#     def __init__(self, num_entities, num_relations, num_classes, sub_edge_index, sub_edge_type, num_nodes, beta):
#         super(RGCNPerturb, self).__init__()
        
#         #we need edge index and transform it into the adjacency matrix
#         self.sub_edge_index = sub_edge_index
#         self.sub_edge_type = sub_edge_type
#         self.num_nodes = num_nodes
#         self.num_relations = num_relations
#         self.beta = beta
        
#         # Initialize P_vec for perturbation
#         self.P_vec_size = int((len(sub_edge_type)))
#         self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))
#         self.reset_parameters()
#         print(f"P VECTOR !",self.P_vec)

#         # Original RGCN layers with frozen weights
#         self.rgcn = RGCN(num_entities, num_relations, num_classes)
    
#     def reset_parameters(self, eps=10**-4):
#         with torch.no_grad():
#             torch.sub(self.P_vec, eps)

#     def forward(self, sub_edge_index, sub_edge_type):
#         self.sub_edge_index = sub_edge_index
#         self.sub_edge_type = sub_edge_type
#         # Convert P_vec to a perturbed edge index or edge type

#         """
#         self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)      # Ensure symmetry
#         A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
#         A_tilde.requires_grad = True
#         A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj + torch.eye(self.num_nodes)  

#         D_tilde = get_degree_matrix(A_tilde).detach()       # Don't need gradient of this
# 		# Raise to power -1/2, set all infs to 0s
#         D_tilde_exp = D_tilde ** (-1 / 2)
#         D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

# 		# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
#         norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp) """

#         """ row, col, edge_attr = A_tilde.t().coo()
#         perturbed_edge_index = torch.stack([row, col], dim=0)

#         #perturbed edge_type according to the perturbed edge index
#         #we first need to get the edge remorved from P and then we can get the new edge type
#         perturbed_edge_type = self.edge_type[perturbed_edge_index[0], perturbed_edge_index[1]]

#         # Create a mask for edges that are retained in the perturbed_edge_index
#         mask = torch.zeros_like(self.edge_index[0], dtype=torch.bool)
#         for i in range(perturbed_edge_index.size(1)):
#             mask |= (self.edge_index[0] == perturbed_edge_index[0, i]) & (self.edge_index[1] == perturbed_edge_index[1, i])

#         # Apply the mask to filter the edge_types accordingly
#         perturbed_edge_type = self.edge_type[mask]
#         """

#         self.mask = (torch.sigmoid(self.P_vec))  # Apply sigmoid to P_vec to ensure values are in (0, 1) range
#         print(self.mask)
#         self.mask = self.mask.long()
#         print(self.mask)

#         # Apply the mask to edge_index
#         perturbed_edge_index = self.sub_edge_index[:, self.mask]

#         # Apply the mask to edge_type
#         perturbed_edge_type = self.sub_edge_type[self.mask]
  
#         # Ensure perturbed_edge_index and perturbed_edge_type are aligned and have the same length
#         assert perturbed_edge_index.size(1) == len(perturbed_edge_type), "Mismatch between perturbed_edge_index and perturbed_edge_type lengths"

#         out = self.rgcn.forward(perturbed_edge_index, perturbed_edge_type)
#         return out
    
#     def forward_prediction(self):
#         # self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()      # threshold P_hat
#         self.mask_ = (torch.sigmoid(self.P_vec) > 0.5).float()  # Create a binary mask

#         # Apply the mask to edge_index
#         perturbed_edge_index = self.sub_edge_index[:, self.mask_.bool()]

#         # Apply the mask to edge_type
#         perturbed_edge_type = self.sub_edge_type[self.mask_.bool()]

#         # Ensure perturbed_edge_index and perturbed_edge_type are aligned and have the same length
#         assert perturbed_edge_index.size(1) == len(perturbed_edge_type), "Mismatch between perturbed_edge_index and perturbed_edge_type lengths"
#         print(f"both index are equal : {perturbed_edge_index.size(1)} and {len(perturbed_edge_type)}")

#         out = self.rgcn.forward(perturbed_edge_index, perturbed_edge_type)
#         return out, self.mask_

#     def loss(self, output, y_pred_orig, y_pred_new_actual):
#         pred_same = (y_pred_new_actual == y_pred_orig).float()  
#         print(pred_same)

#         # output = output.unsqueeze(0)
#         print(f"sub_edge_index : {self.sub_edge_index}")
#         print(f"sub_edge_type : {self.sub_edge_type}")
#         cf_edge_index = self.sub_edge_index[:, self.mask_]
#         cf_edge_type = self.sub_edge_type[self.mask_]
#         cf_edge_index.requires_grad = True
#         # cf_edge_type.requires_grad = True

#         print(f"y pred original : {y_pred_orig}")
#         print(f"p_vec : {self.P_vec}")
#         print(f"P_vec length : {len(self.P_vec)}")
#         print(f"the mask partly {self.mask_}")
#         print(f"cf_edge_index : {cf_edge_index}")
#         print(f"cf_edge_type : {cf_edge_type}")
#         loss_pred = - F.nll_loss(output, y_pred_orig)
#         loss_graph_dist = sum(sum(abs(cf_edge_index - self.sub_edge_index))) #/ 2      # Number of edges changed (symmetrical)

#         if torch.max(self.P_vec) > 1.05:
#             regularization_term = 0.1 * (torch.max(self.P_vec) - 1)
#             # time.sleep(10)
#         else :
#             regularization_term = 0
#         loss_total = (-pred_same) * loss_pred + self.beta * loss_graph_dist
#         loss_total = loss_total + regularization_term
#         return loss_total, loss_pred, loss_graph_dist, cf_edge_index, cf_edge_type

class RGCNPerturb(nn.Module):
    def __init__(self, num_entities, num_relations, num_classes, sub_edge_index, sub_edge_type, num_nodes, beta):
        super(RGCNPerturb, self).__init__()
        
        self.sub_edge_index = sub_edge_index
        self.sub_edge_type = sub_edge_type
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.beta = beta
        
        # Initialize P_vec for perturbation
        self.P_vec_size = int((len(sub_edge_type)))
        self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size))) # Initialized uniformly for a range, can be adjusted
        # self.reset_parameters()
        #require gradients
        self.P_vec.requires_grad = True

        # Original RGCN layers with frozen weights
        self.rgcn = RGCN(num_entities, num_relations, num_classes)
    
    def reset_parameters(self, eps=10**-2):
        # Think more about how to initialize this
        with torch.no_grad():
            torch.sub(self.P_vec, eps)
    
    def forward(self, sub_edge_index, sub_edge_type):
        # Here, for simplicity, we keep the original edges and types unchanged in the forward pass
        # Assuming the perturbation process is handled externally or in forward_prediction
        self.sub_edge_index = sub_edge_index
        self.sub_edge_type = sub_edge_type

        """ temperature = 0.5  # Temperature for Gumbel-Softmax, adjust based on your needs
        num_edges = self.P_vec.size(0)
        num_relations = len(sub_edge_type) +1  # Including the new relation type for self-loops

        # Generate soft selections
        gumbel_softmax = RelaxedOneHotCategorical(temperature, logits=self.P_vec)
        soft_selection = gumbel_softmax.rsample()

        # Conceptual step for soft transformation
        # This requires your edge_type tensor to be in a format where you can apply a continuous operation
        # Here, we assume edge_type is a one-hot encoded tensor of shape (num_edges, num_relations)
        one_hot_edge_types = F.one_hot(sub_edge_type, num_relations)
        new_relation_vector = torch.zeros(num_relations)
        new_relation_vector[num_relations - 1] = 1  # Index 90 corresponds to the new relation type, assuming 0-based indexing

        # Softly update the edge types based on the selection probabilities
        soft_updated_edge_types = one_hot_edge_types * (1 - soft_selection.unsqueeze(-1)) + new_relation_vector * soft_selection.unsqueeze(-1)
        """
        #element wise multiplication to either 0  or 1
        perturbed_edge_type = sub_edge_type * (self.P_vec) + 91 * (1 - self.P_vec)
        print("P_vec : ", self.P_vec)
        print(f"sub_edge_index : {sub_edge_index}")
        print(f"sub_edge_type : {perturbed_edge_type}")

        out = self.rgcn(sub_edge_index, perturbed_edge_type)
        return out
    
    def forward_prediction(self):
        # Assuming P_vec directly influences whether to convert an edge to a self-loop
        # For simplicity, let's threshold P_vec to decide which edges to perturb, but note this is non-differentiable
        """ threshold = 0.5  # This threshold can be adjusted
        mask = self.P_vec > threshold
        perturbed_edge_index = self.sub_edge_index.clone()
        perturbed_edge_type = self.sub_edge_type.clone()

        # Applying the mask for self-loop conversion
        for i, to_perturb in enumerate(mask):
            if to_perturb:
                perturbed_edge_index[1, i] = perturbed_edge_index[0, i]  # Convert to self-loop
                perturbed_edge_type[i] = 91  # Change to new relation type

        self.p1 = perturbed_edge_index
        self.p2 = perturbed_edge_type 
        out = self.rgcn(perturbed_edge_index, perturbed_edge_type)"""
         #element wise multiplication to either 0  or 1
        perturbed_edge_type = self.sub_edge_type * (1 - self.P_vec) + 91 * self.P_vec
        print(f"sub_edge_index : {self.sub_edge_index}")
        print(f"sub_edge_type : {perturbed_edge_type}")

        out = self.rgcn(self.sub_edge_index, perturbed_edge_type)

        return out, self.P_vec
    

    def loss(self, output, y_pred_orig, y_pred_new_actual):
        # Assuming y_pred_orig and y_pred_new_actual are tensors of class indices
        pred_diff = F.nll_loss(output, y_pred_orig, reduction='mean')  # Calculate prediction difference
        mask = self.P_vec > 0.5  # Reusing the thresholding logic from forward_prediction

        # Count the number of edges changed in a differentiable mannear
        num_edges_changed = torch.sum(mask.float())  # Simple count of changed edges
        loss_graph_dist = num_edges_changed  # Simple count of changed edges as graph distance

        regularization_term = torch.max(self.P_vec) - 1 if torch.max(self.P_vec) > 1 else torch.tensor(0.0)

        loss_total = pred_diff + self.beta * loss_graph_dist + regularization_term
        return loss_total, pred_diff, loss_graph_dist, self.p1, self.p2

class CFExplainer:
    """
    CF Explainer class, returns counterfactual subgraph
    """
    def __init__(self, model, sub_adj, sub_feat, n_hid, dropout,
                  sub_labels, y_pred_orig, num_classes, beta, device, model_type):
        super(CFExplainer, self).__init__()
        self.model = model
        self.model.eval()
        self.sub_adj = sub_adj
        self.sub_feat = sub_feat
        self.n_hid = n_hid
        self.dropout = dropout
        self.sub_labels = sub_labels
        self.y_pred_orig = y_pred_orig
        self.beta = beta
        self.num_classes = num_classes
        self.device = device
        self.model_type = model_type

        print(sub_adj.shape)
        if self.model_type == "GCN":
            self.cf_model = GCNPerturb(self.sub_feat.shape[1], n_hid, self.num_classes, self.sub_adj, dropout, beta)
        if self.model_type == "GCN2Layer":
            self.cf_model = GCN2LayerPerturb(self.sub_feat.shape[1], n_hid, n_hid,
                                                self.num_classes, self.sub_adj, dropout, beta)
        elif self.model_type == "synthetic":
            self.cf_model = GCNSyntheticPerturb(self.sub_feat.shape[1], n_hid, n_hid,
                                                self.num_classes, self.sub_adj, dropout, beta)
        elif self.model_type == "GCNSyntheticPerturb_synthetic":
            print("GOOD JOB")
            self.cf_model = GCNSyntheticPerturb_synthetic(self.sub_feat.shape[1], n_hid, n_hid,
                                                self.num_classes, self.sub_adj, dropout, beta, edge_additions=True)
        

        print("Accessing the cf model state_dict")
        # the required architecture for the perturbed model
        for values in self.cf_model.state_dict():
            print(values, "\t", self.cf_model.state_dict()[values].size())
        # the architecture how it leaves from the PyG model
        print("Accessing the model state_dict")
        for values in self.model.state_dict():
            print(values, "\t", self.model.state_dict()[values].size())
        # we change the names of the keys so that the weights and biases can be transferred from model to cf_model
        corrected_dict = self.model.state_dict()
        # print(corrected_dict.keys())
        if 'gc1.lin.weight' in corrected_dict:
            corrected_dict['gc1.weight'] = torch.transpose(corrected_dict['gc1.lin.weight'],0,1)
        if 'gc2.lin.weight' in corrected_dict:
            corrected_dict['gc2.weight'] = torch.transpose(corrected_dict['gc2.lin.weight'],0,1)
        if 'gc3.lin.weight' in corrected_dict:
            corrected_dict['gc3.weight'] = torch.transpose(corrected_dict['gc3.lin.weight'],0,1)
        if self.model_type == "GCN":
            if 'conv1.W' in corrected_dict:
                corrected_dict['conv1.weight'] = corrected_dict['conv1.W']
                #delete conv1.W
                del corrected_dict['conv1.W']
            if 'conv2.W' in corrected_dict:
                corrected_dict['conv2.weight'] = corrected_dict['conv2.W']
                #delete conv2.W
                del corrected_dict['conv2.W']
            if 'conv3.W' in corrected_dict:
                corrected_dict['conv3.weight'] = corrected_dict['conv3.W']
                #delete conv3.W
                del corrected_dict['conv3.W']
            
        
        # print(corrected_dict.keys())
        # print(self.model.state_dict())
        self.cf_model.load_state_dict(corrected_dict, strict=False)
        # print(self.cf_model.state_dict())

        # the architecture ready for perturbations
        for values in self.model.state_dict():
            print(values, "\t", self.model.state_dict()[values].size())

        # Freeze weights from original model in cf_model
        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            print("orig model requires_grad: ", name, param.requires_grad)
        for name, param in self.cf_model.named_parameters():
            print("cf model requires_grad: ", name, param.requires_grad)
        time.sleep(10)
        

    def explain(self, cf_optimizer, node_idx, new_idx, lr, n_momentum, num_epochs):
        self.node_idx = node_idx
        self.new_idx = new_idx

        self.x = self.sub_feat
        self.A_x = self.sub_adj
        self.D_x = get_degree_matrix(self.A_x)

        #check that new model gives same prediction on full graph and subgraph
        print(self.model)
        print(self.cf_model)
        # time.sleep(10)
        print('Check that NEW model gives same prediction on full graph and subgraph')
        print(f"Original prediction for node {self.node_idx} is {self.y_pred_orig}")
        print(f"Original prediction for node {self.node_idx} is {self.model(self.x, (self.A_x).to_dense().to_sparse())}")
        print(f" forward for node {self.node_idx} is {self.cf_model.forward(self.x, self.A_x)}")
        print(f" forward prediction for node {self.node_idx} is {self.cf_model.forward_prediction(self.x)}")
        output_actual, self.P = self.cf_model.forward_prediction(self.x)
        print(f" class predictions for node {self.node_idx} is {self.y_pred_orig, torch.argmax(self.cf_model.forward(self.x, self.A_x)[self.new_idx]), torch.argmax(output_actual[self.new_idx])}")
        
        # time.sleep(10)

        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr, nesterov=False, momentum=0.0)
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "adam":
            self.cf_optimizer = optim.Adam(self.cf_model.parameters(), lr=lr)
        print("HERE:")
        # print(self.cf_model.parameters())

        print('Check if predictions change with optimizer ???')
        print(f"Original prediction for node {self.node_idx} is {self.y_pred_orig}")
        print(f"Original prediction for node {self.node_idx} is {self.model(self.x, (self.A_x).to_dense().to_sparse())}")
        print(f" forward for node {self.node_idx} is {self.cf_model.forward(self.x, self.A_x)}")
        print(f" forward prediction for node {self.node_idx} is {self.cf_model.forward_prediction(self.x)}")
        output_actual, self.P = self.cf_model.forward_prediction(self.x)
        print(f" class predictions for node {self.node_idx} is {self.y_pred_orig, torch.argmax(self.cf_model.forward(self.x, self.A_x)[self.new_idx]), torch.argmax(output_actual[self.new_idx])}")
        
        
        best_cf_examples = []
        best_loss_measure = 199999.0
        num_cf_examples = 0

        
        for epoches in range(num_epochs):
            print("Epoch: ", epoches)
            new_example, loss_total = self.train(epoches)
            if new_example != [] and loss_total < best_loss_measure and loss_total != 0.0:
                best_cf_examples.append(new_example)
                print("New best loss: ", loss_total)
                print("Best loss: ", best_loss_measure)
                print("Best CF example: ", best_cf_examples)
                best_loss_measure = 999999.0
                
                best_loss_measure = loss_total

                print("Best loss: ", best_loss_measure)
                num_cf_examples += 1
                print("Number of CF examples: ", num_cf_examples)
                time.sleep(4)
            
            #pause when exmaple 2
            if epoches == 2:
                # time.sleep(80)
                print(" epoch 2")
            if epoches == 10:
                # time.sleep(80)
                print(" epoch 10")
            if best_loss_measure == 199999.0 and num_cf_examples > 0:
                print("No CF example found")
                # time.sleep(15)
            if epoches == 99:
                print("epoch 99")
                print("Best loss: ", best_loss_measure)
                print("Number of CF examples: ", num_cf_examples)
                time.sleep(1)
        
        print('end of training')
        time.sleep(1)
        
        print("{} CF examples for node_idx = {}".format(num_cf_examples, self.node_idx))
        time.sleep(1)
        return(best_cf_examples)


    def train(self, epoch):
        print("Training")
        t = time.time()
        

        # output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
        # output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
        output = self.cf_model.forward(self.x, self.A_x)
        output_actual, _ = self.cf_model.forward_prediction(self.x)
        #train after the forward pass in order to have the same prediction for the subgraph and the full graph

        # print("output",output)
        # print("output actual",output_actual)

        print(output[self.new_idx])
        print(output_actual[self.new_idx])

        self.cf_model.train()
        self.cf_optimizer.zero_grad()

        output_1 = self.cf_model.forward(self.x, self.A_x)
        output_actual_1, _ = self.cf_model.forward_prediction(self.x)
        #train after the forward pass in order to have the same prediction for the subgraph and the full graph

        # print("output",output)
        # print("output actual",output_actual)

        print(output_1[self.new_idx])
        print(output_actual_1[self.new_idx])
        # time.sleep(10)

        # Need to use new_idx from now on since sub_adj is reindexed
        y_pred_new = torch.argmax(output[self.new_idx])
        y_pred_new_actual = torch.argmax(output_actual[self.new_idx])
        print("y_pred_new",y_pred_new)
        print("y_pred_new_actual",y_pred_new_actual)
        print("y_pred_orig",self.y_pred_orig)
        # time.sleep(10)
        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        loss_total, loss_pred, loss_graph_dist, cf_adj = self.cf_model.loss(output[self.new_idx], self.y_pred_orig, y_pred_new_actual)
        print("total loss", loss_total)
        # graph = make_dot(loss_total, params=dict(self.cf_model.named_parameters()))
        # graph.render("computational_graph_before", format="png")  # Saves the graph to a file named "computational_graph.png"
        print("gradients of P", self.cf_model.P_vec.grad)
        # time.sleep(10)
        loss_total.backward()
        clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()
        # graph = make_dot(loss_total, params=dict(self.cf_model.named_parameters()))
        # graph.render("computational_graph_after", format="png")  # Saves the graph to a file named "computational_graph.png"
        
        print(f"gradients of P", self.cf_model.P_vec.grad)
        print('Node idx: {}'.format(self.node_idx),
              'New idx: {}'.format(self.new_idx),
              'Epoch: {:04d}'.format(epoch + 1),
              'loss: {:.4f}'.format(loss_total.item()),
              'pred loss: {:.4f}'.format(loss_pred.item()),
              'graph loss: {:.4f}'.format(loss_graph_dist.item()))
        print('Output: {}\n'.format(output[self.new_idx].data),
              'Output nondiff: {}\n'.format(output_actual[self.new_idx].data),
              'orig pred: {}, new pred: {}, new pred nondiff: {}'.format(self.y_pred_orig, y_pred_new, y_pred_new_actual))
        print(" ")
        cf_stats = []
        print(y_pred_new)
        print(y_pred_new_actual)
        print("__________________")
        if y_pred_new_actual != self.y_pred_orig: #and loss_total.item() > 0 this part was added to make sure edges are removed
            cf_stats = [self.node_idx, self.new_idx,
                        cf_adj, self.sub_adj,
                        self.y_pred_orig, y_pred_new,
                        y_pred_new_actual, self.sub_labels[self.new_idx],
                        self.sub_adj.shape[0], loss_total,
                        loss_pred.item(), loss_graph_dist]

        return(cf_stats, loss_total.item())
    
class R_CFExplainer:
    """
    CF Explainer class, returns counterfactual subgraph
    """
    def __init__(self, model, num_entities, num_relations, num_classes, sub_edge_index, sub_edge_type, num_nodes, y_pred_orig, dropout, beta, model_type):
        super(R_CFExplainer, self).__init__()
        self.model = model
        self.model.eval()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_classes = num_classes
        self.sub_edge_index = sub_edge_index
        self.sub_edge_type = sub_edge_type
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.y_pred_orig = y_pred_orig
        self.beta = beta
        self.num_classes = num_classes
        self.model_type = model_type

        if self.model_type == "RGCN":
            self.cf_model = RGCNPerturb(num_entities, num_relations, num_classes, sub_edge_index, sub_edge_type, num_nodes, beta)
        
        print(f"CF model : {self.cf_model}")
        print("Accessing the cf model state_dict")
        # the required architecture for the perturbed model
        for values in self.cf_model.state_dict():
            print(values, "\t", self.cf_model.state_dict()[values].size())
        # the architecture how it leaves from the PyG model
        print("Accessing the model state_dict")
        for values in self.model.state_dict():
            print(values, "\t", self.model.state_dict()[values].size())

        # we change the names of the keys so that the weights and biases can be transferred from model to cf_model
        corrected_dict = self.model.state_dict()
        # print(corrected_dict.keys())
        
        if self.model_type == "RGCN":
            #addrgcn. in front of each key
            for key in list(corrected_dict.keys()):
                corrected_dict['rgcn.'+key] = corrected_dict[key]
                del corrected_dict[key]
            
        
        print(corrected_dict.keys())
        # print(self.model.state_dict())
        self.cf_model.load_state_dict(corrected_dict, strict=False)
        time.sleep(10)
        # print(self.cf_model.state_dict())

        # the architecture ready for perturbations
        for values in self.model.state_dict():
            print(values, "\t", self.model.state_dict()[values].size())

        # Freeze weights from original model in cf_model
        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias") or name.endswith("root"):
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            print("orig model requires_grad: ", name, param.requires_grad)
        for name, param in self.cf_model.named_parameters():
            print("cf model requires_grad: ", name, param.requires_grad)
        time.sleep(10)
        

    def explain(self, cf_optimizer, node_idx, new_idx, lr, n_momentum, num_epochs):
        self.node_idx = node_idx
        self.new_idx = new_idx

        #check that new model gives same prediction on full graph and subgraph
        print(self.model)
        print(self.cf_model)
        # time.sleep(10)
        print('Check that NEW model gives same prediction on full graph and subgraph')
        print(f"Original prediction for node {self.node_idx} is {self.y_pred_orig}")
        print(f"Original prediction for node {self.node_idx} is {self.model(self.sub_edge_index, self.sub_edge_type)}")
        print(f" forward for node {self.node_idx} is {self.cf_model.forward(self.sub_edge_index, self.sub_edge_type)}")
        print(f" forward prediction for node {self.node_idx} is {self.cf_model.forward_prediction()}")
        output_actual, self.P = self.cf_model.forward_prediction()
        print(f" class predictions for node {self.node_idx} is {self.y_pred_orig, torch.argmax(self.cf_model.forward(self.sub_edge_index, self.sub_edge_type)[self.new_idx]), torch.argmax(output_actual[self.new_idx])}")
        
        time.sleep(10)

        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr, nesterov=False, momentum=0.0)
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "adam":
            self.cf_optimizer = optim.Adam(self.cf_model.parameters(), lr=lr)
        print("HERE:")
        # print(self.cf_model.parameters())
        
        best_cf_examples = []
        best_loss_measure = 199999.0
        num_cf_examples = 0
        
        for epoches in range(num_epochs):
            print("Epoch: ", epoches)
            new_example, loss_total = self.train(epoches)
            if new_example != [] and loss_total < best_loss_measure and loss_total != 0.0:
                best_cf_examples.append(new_example)
                print("New best loss: ", loss_total)
                print("Best loss: ", best_loss_measure)
                print("Best CF example: ", best_cf_examples)
                best_loss_measure = 999999.0
                
                best_loss_measure = loss_total

                print("Best loss: ", best_loss_measure)
                num_cf_examples += 1
                print("Number of CF examples: ", num_cf_examples)
                time.sleep(4)
            
            #pause when exmaple 2
            if epoches == 2:
                # time.sleep(80)
                print(" epoch 2")
            if epoches == 10:
                # time.sleep(80)
                print(" epoch 10")
            if best_loss_measure == 199999.0 and num_cf_examples > 0:
                print("No CF example found")
                # time.sleep(15)
            if epoches == 99:
                print("epoch 99")
                print("Best loss: ", best_loss_measure)
                print("Number of CF examples: ", num_cf_examples)
                time.sleep(1)
        
        print('end of training')
        time.sleep(1)
        
        print("{} CF examples for node_idx = {}".format(num_cf_examples, self.node_idx))
        time.sleep(1)
        return(best_cf_examples)


    def train(self, epoch):
        print("Training")
        t = time.time()
        

        # output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
        # output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
        output = self.cf_model.forward(self.sub_edge_index, self.sub_edge_type)
        output_actual, _ = self.cf_model.forward_prediction()
        #train after the forward pass in order to have the same prediction for the subgraph and the full graph

        # print("output",output)
        # print("output actual",output_actual)

        print(output[self.new_idx])
        print(output_actual[self.new_idx])

        self.cf_model.train()
        self.cf_optimizer.zero_grad()

        output_1 = self.cf_model.forward(self.sub_edge_index, self.sub_edge_type)
        output_actual_1, _ = self.cf_model.forward_prediction()
        #train after the forward pass in order to have the same prediction for the subgraph and the full graph

        # print("output",output)
        # print("output actual",output_actual)

        print(output_1[self.new_idx])
        print(output_actual_1[self.new_idx])
        # time.sleep(10)

        # Need to use new_idx from now on since sub_adj is reindexed
        y_pred_new = torch.argmax(output[self.new_idx])
        y_pred_new_actual = torch.argmax(output_actual[self.new_idx])
        print("y_pred_new",y_pred_new)
        print("y_pred_new_actual",y_pred_new_actual)
        print("y_pred_orig",self.y_pred_orig)
        # time.sleep(10)
        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        loss_total, loss_pred, loss_graph_dist, cf_edge_index, cf_edge_type = self.cf_model.loss(output[self.new_idx], self.y_pred_orig, y_pred_new_actual)
        print("total loss", loss_total)
        graph = make_dot(loss_total, params=dict(self.cf_model.named_parameters()))
        graph.render("computational_graph_rgcn_before", format="png")  # Saves the graph to a file named "computational_graph.png"
        print("gradients of P", self.cf_model.P_vec.grad)
        # time.sleep(10)
        loss_total.backward()
        clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()

        
        # After the optimizer step, binarize P_vec
        with torch.no_grad():  # Ensure we don't track this operation
            for param in self.cf_model.parameters():
                if param is self.cf_model.P_vec:  # Identifying the specific parameter P_vec
                    param.data = (param.data > 0.8).float()  # Set values to 0 or 1 based on a threshold

        graph = make_dot(loss_total, params=dict(self.cf_model.named_parameters()))
        graph.render("computational_graph_rgcn_after", format="png")  # Saves the graph to a file named "computational_graph.png"
        
        print(f"gradients of P", self.cf_model.P_vec.grad)
        print('Node idx: {}'.format(self.node_idx),
            'New idx: {}'.format(self.new_idx),
            'Epoch: {:04d}'.format(epoch + 1),
            'loss: {:.4f}'.format(loss_total.item()),
            'pred loss: {:.4f}'.format(loss_pred.item()),
            'graph loss: {:.4f}'.format(loss_graph_dist.item()))
        print('Output: {}\n'.format(output[self.new_idx].data),
            'Output nondiff: {}\n'.format(output_actual[self.new_idx].data),
            'orig pred: {}, new pred: {}, new pred nondiff: {}'.format(self.y_pred_orig, y_pred_new, y_pred_new_actual))
        print(" ")
        cf_stats = []
        print(y_pred_new)
        print(y_pred_new_actual)
        print("__________________")
        if y_pred_new_actual != self.y_pred_orig: #and loss_total.item() > 0 this part was added to make sure edges are removed
            cf_stats = [self.node_idx, self.new_idx,
                        cf_edge_index, cf_edge_type,
                        self.y_pred_orig, y_pred_new,
                        y_pred_new_actual, 
                        self.sub_adj.shape[0], loss_total,
                        loss_pred.item(), loss_graph_dist]

        return(cf_stats, loss_total.item())
