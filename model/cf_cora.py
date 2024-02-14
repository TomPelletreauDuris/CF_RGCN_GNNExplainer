import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn.parameter import Parameter
import time
import torch.optim as optim
from model.GCN import GraphConvolution, GraphConvolutionLayer, RGCN

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
        if bias is True:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)


    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is True:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
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
        self.conv2 = GraphConvolutionPerturb(nhid, nclass, bias=False)
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
        # print(f"Dropout in PERTURB : {x}")
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
        # t1 = [-8.0272, -7.9911, -3.1536, -1.3117, -0.8077, -1.4214]
        # t1 = torch.tensor(t1)
        # t1 = t1.unsqueeze(0)
        # t2 = [4]
        # t2 = torch.tensor(t2)
        # # t2 = t2.unsqueeze(0)
        # loss_pred = - F.nll_loss(t1, t2)
        # print(f"loss_pred : {loss_pred}")
        loss_pred = - F.nll_loss(output, y_pred_orig)
        print(f"check the nature of adj before loss : {self.adj}")
        loss_graph_dist = sum(sum(abs(cf_adj - self.adj)))  #/ 2      # Number of edges changed (symmetrical)

        #if max p_vex is > 1.1 let's put a penalty
        if torch.max(self.P_vec) > 1.01:
            regularization_term = 1.5 * (torch.max(self.P_vec) - 1)
            # time.sleep(10)
        else :
            regularization_term = 0.0

        #if all values of P_vec are ~1, let's put a penalty
        if torch.max(self.P_vec) < 1.01 and torch.min(self.P_vec) > 0.9:
            regularization_term_2 = 1.5 * (torch.max(self.P_vec) - 0.9)
            # time.sleep(10)
        else :
            regularization_term_2 = 0.0

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist + regularization_term + regularization_term_2
        return loss_total, loss_pred, loss_graph_dist, cf_adj


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
        print(f" class predictions for node {self.node_idx} is {torch.argmax(self.y_pred_orig), torch.argmax(self.cf_model.forward(self.x, self.A_x)[self.new_idx]), torch.argmax(output_actual[self.new_idx])}")
        
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

        print('Check if predictions change with optimizer ???')
        print(f"Original prediction for node {self.node_idx} is {self.y_pred_orig}")
        print(f"Original prediction for node {self.node_idx} is {self.model(self.x, (self.A_x).to_dense().to_sparse())}")
        print(f" forward for node {self.node_idx} is {self.cf_model.forward(self.x, self.A_x)}")
        print(f" forward prediction for node {self.node_idx} is {self.cf_model.forward_prediction(self.x)}")
        output_actual, self.P = self.cf_model.forward_prediction(self.x)
        print(f" class predictions for node {self.node_idx} is {torch.argmax(self.y_pred_orig), torch.argmax(self.cf_model.forward(self.x, self.A_x)[self.new_idx]), torch.argmax(output_actual[self.new_idx])}")
        
        best_cf_examples = []
        best_loss_measure = 199999.0
        num_cf_examples = 0

        for epoches in range(num_epochs):
            print("Epoch: ", epoches)
            new_example, loss_total = self.train(epoches)
            if new_example != [] and loss_total < best_loss_measure: #and loss_total != 0.0:
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
        # time.sleep(1)
        
        print("{} CF examples for node_idx = {}".format(num_cf_examples, self.node_idx))
        # time.sleep(1)
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

        # output_1 = self.cf_model.forward(self.x, self.A_x)
        # output_actual_1, _ = self.cf_model.forward_prediction(self.x)
        #train after the forward pass in order to have the same prediction for the subgraph and the full graph

        # print("output",output)
        # print("output actual",output_actual)

        # print(output_1[self.new_idx])
        # print(output_actual_1[self.new_idx])
        # time.sleep(10)

        # Need to use new_idx from now on since sub_adj is reindexed
        y_pred_new = torch.argmax(output[self.new_idx])
        y_pred_new_actual = torch.argmax(output_actual[self.new_idx])
        print("y_pred_new",y_pred_new)
        print("y_pred_new_actual",y_pred_new_actual)
        print("y_pred_orig",torch.argmax(self.y_pred_orig))
        # time.sleep(10)
        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        loss_total, loss_pred, loss_graph_dist, cf_adj = self.cf_model.loss(y_pred_new, torch.argmax(self.y_pred_orig), y_pred_new_actual)
        print("total loss", loss_total)
        # graph = make_dot(loss_total, params=dict(self.cf_model.named_parameters()))
        # graph.render("computational_graph_before", format="png")  # Saves the graph to a file named "computational_graph.png"
        print("gradients of P", self.cf_model.P_vec.grad)
        # time.sleep(10)
        loss_total.backward()
        clip_grad_norm_(self.cf_model.parameters(), 2.0)
        
        """ if self.cf_model.P_vec.grad is not None:
            # Use torch.where to selectively apply changes
            self.cf_model.P_vec.grad = torch.where(self.cf_model.P_vec.grad > 0, -self.cf_model.P_vec.grad, self.cf_model.P_vec.grad)
        """
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
        if y_pred_new_actual != torch.argmax(self.y_pred_orig) and loss_graph_dist.item() > 0: # this part was added to make sure edges are removed
            cf_stats = [self.node_idx, self.new_idx,
                        cf_adj, self.sub_adj,
                        torch.argmax(self.y_pred_orig), y_pred_new,
                        y_pred_new_actual, self.sub_labels[self.new_idx],
                        self.sub_adj.shape[0], loss_total,
                        loss_pred.item(), loss_graph_dist.item()]

        return(cf_stats, loss_total.item())
