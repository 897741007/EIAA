import torch
from torch import nn
import numpy as np
from Smi2Graph import SMI_grapher
from time import time
import os

class GAT_predictor(nn.Module):
    def __init__(self, hidden_dim, layer_num, head_num, dict_size, dropout=0, bond_influence=0, prediction_class=2, device='cuda'):
        # param : bond_influence   --> how to merge the influence of bond in the attention
        #                              0: ingore the influence of bond
        #                              1: add the embedding of bond to K
        #                              2: mul the embedding of bond to K
        # param : prediction_class --> the number of classification labels
        #                              1: the task is a regression task
        #                              n(n>1): the task is a classification task 
        super(GAT_predictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.head_num = head_num
        self.dict_size = dict_size
        assert dropout < 1
        self.dropout = dropout
        assert bond_influence in (0, 1, 2)
        self.bond_influence = bond_influence
        self.prediction_class = prediction_class
        self.scale = np.sqrt(hidden_dim)
        self.sp_dim = int(hidden_dim/head_num)
        self.device = device
        self.GAT_init()
    
    def GAT_init(self):
        self.atom_embedding_layer = nn.Embedding(self.dict_size, self.hidden_dim).to(self.device)
        # bond type: non-link, self-link, single-bond, double-bond, trible-bond, π-adj, π-meta, π-para
        if self.bond_influence:
            # N(0, 1), if the influence of bond is add to the weight matrix, the distribution of embedding ~ N(0,1)
            bond_embedding_weight = torch.randn(8, self.hidden_dim, device=self.device)
            if self.bond_influence==2:
                # N(1, 1), if the influence of bond is mul to the weight matrix, the distribution of embedding ~ N(1,1)
                bond_embedding_weight = bond_embedding_weight + torch.ones_like(bond_embedding_weight, device=self.device)
            self.bond_embedding_layer = nn.Embedding(8, self.hidden_dim,_weight=bond_embedding_weight).to(self.device)

        self.q_layers = nn.ModuleList()
        self.k_layers = nn.ModuleList()
        self.v_layers = nn.ModuleList()
        self.ew_layers = nn.ModuleList()
        self.stack_head_layers = nn.ModuleList()
        self.FNN_layers = nn.ModuleList()
        for _ in range(self.layer_num):
            self.q_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device))
            self.k_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device))
            self.v_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device))
            self.ew_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device))
            self.stack_head_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device))
            self.FNN_layers.append(nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim*4, bias=False).to(self.device), 
                                                  nn.Linear(self.hidden_dim*4, self.hidden_dim, bias=False).to(self.device)]))
        self.output_layer = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device),
                                          nn.Tanh().to(self.device))
        self.predictor_layer = nn.Linear(self.hidden_dim, self.prediction_class).to(self.device)
        if self.prediction_class > 1:
            self.predictor_softmax = nn.Softmax(dim=-1)

    def weighted_qkmm(self, q, k, w):
        # q.shape : [batch_size, head_num, token_len, split_dim]
        # k.shape : [batch_size, head_num, token_len, split_dim]
        # w.shape : [batch_size, head_num, token_len, token_len, split_dim]
        q = q.unsqueeze(-2).expand(w.shape)
        k = k.unsqueeze(-3).expand(w.shape)
        if self.bond_influence == 1:
            k = k + w
        else:
            k = torch.mul(k, w)
        sim_s = torch.mul(q, k)
        sim = sim_s.sum(-1)
        return sim
    
    def edge_attention(self, attn_out, w):
        # update the feature of edge according to edge and the linked atoms
        # new_edge : sum(Softmax(edge*atom_0, edge*edge, edge*atom_1) * (atom_0, edge, atom_1))
        # atten_out.shape : [batch_size, token_len, hidden_dim]
        # w.shape : [batch_size, token_len, token_len, hidden_dim]
        ew_0 = w.unsqueeze(-2)
        ew_1 = w.unsqueeze(-1)
        ew_s = torch.matmul(ew_0, ew_1).squeeze(-2)
        # atom_0 * edge
        aw_0 = attn_out.unsqueeze(2).expand_as(w).unsqueeze(-1)
        aw_0s = torch.matmul(ew_0, aw_0).squeeze(-2)
        # atom_1 * edge
        aw_1 = attn_out.unsqueeze(1).expand_as(w).unsqueeze(-1)
        aw_1s = torch.matmul(ew_0, aw_1).squeeze(-2)
        # Softmax
        weight = torch.cat((ew_s, aw_0s, aw_1s), dim=-1)
        weight = weight/self.scale
        weight = nn.Softmax(dim=-1)(weight).unsqueeze(-2)
        # weighted sum
        hidden = torch.cat((ew_1, aw_0, aw_1), dim=-1).permute((0,1,2,4,3))
        new_e = torch.matmul(weight, hidden).squeeze(-2)
        return new_e

    def split_head(self, tensor):
        # split the head into n heads
        k = tensor.shape
        if len(k) == 4:
            return tensor
        else:
            split_tensor = tensor.reshape(k[0], k[1], self.head_num, self.sp_dim)
            split_tensor = split_tensor.permute(0,2,1,3)
            return split_tensor
    
    def split_bond_embedding(self, bond_embedding):
        # split the bond embedding into n heads
        # bond_embedding shape : [batch_size, tensor_length, tensor_length, embedding_dim]
        # splited_bond_embedding shape : [batch_size, head_num, tensor_length, tensor_length, splited_dim]
        k = bond_embedding.shape
        split_bond_embedding = bond_embedding.reshape(k[0], k[1], k[2], self.head_num, self.sp_dim)
        split_bond_embedding = split_bond_embedding.permute(0,3,1,2,4)
        return split_bond_embedding

    def combine_head(self, tensor):
        # combine the split heads into one head
        k = tensor.shape
        if len(k) == 3:
            return tensor
        else:
            combine_tensor = tensor.permute(0,2,1,3)
            combine_tensor = combine_tensor.reshape(k[0], k[2], self.hidden_dim)
            return combine_tensor
            
    def attention_mask(self, logits, adjacency_matrix):
        # mask attention weights to control the range of attention
        # each atom can only see the atoms it linked and the atom itself
        # different type of bonds are regard as the same bonds
        multi_head_adjacency_matrix = adjacency_matrix.unsqueeze(1).expand(logits.shape)
        #logits[multi_head_adjacency_matrix<0.5] = -np.inf
        #logits[multi_head_adjacency_matrix<0.5] = -1e9
        attn_scores = torch.zeros_like(multi_head_adjacency_matrix, dtype=torch.float)
        attn_scores[multi_head_adjacency_matrix==0] = -np.inf
        logits = logits + attn_scores
        #torch.save(logits, 'test_logits.pkl')
        return logits
    
    def multi_head_attention_layer(self, q, k, v, idx, bond_embedding, attn_mask_template):
        # scaled dot multi head-attention
        # parameter : attn_mask_template ---> the template of masking the attention
        if self.bond_influence:
            logits = self.weighted_qkmm(q, k, bond_embedding)
        else:
            logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        logits = logits/self.scale
        logits = self.attention_mask(logits, attn_mask_template)
        weights = nn.Softmax(dim=-1)(logits)
        if self.dropout:
            weights = nn.Dropout(self.dropout)(weights)
        #torch.save(weights, 'test_weights.pkl')
        out_pre = torch.matmul(weights, v)
        out = self.combine_head(out_pre)
        out = self.stack_head_layers[idx](out)
        return out

    def FNN(self, idx, input_batch):
        # feed forward layer
        # consists of two linear layers, the activation only occurs after the first linear layer
        fnn_hidden_tensor = self.FNN_layers[idx][0](input_batch)
        fnn_hidden_tensor = nn.ReLU()(fnn_hidden_tensor)
        if self.dropout:
            fnn_hidden_tensor = nn.Dropout(self.dropout)(fnn_hidden_tensor)
        fnn_output = self.FNN_layers[idx][1](fnn_hidden_tensor)
        return fnn_output

    def GAT_layer_0(self, input_batch, adj_m, idx, bond_embedding):
        # attention layer in Graph Attention
        input_batch = nn.LayerNorm(self.hidden_dim).to(self.device)(input_batch)
        q = self.split_head(self.q_layers[idx](input_batch))
        k = self.split_head(self.k_layers[idx](input_batch))
        v = self.split_head(self.v_layers[idx](input_batch))
        ew = None
        if self.bond_influence:
            #ew = self.split_bond_embedding(self.ew_layers[idx](bond_embedding))
            ew = self.split_bond_embedding(bond_embedding)
        attn_out = self.multi_head_attention_layer(q, k, v, idx, ew, adj_m)

        if self.dropout:
            attn_out = nn.Dropout(self.dropout)(attn_out)
        # the first residual
        attn_resid = attn_out + input_batch
        # the first layer normalization
        #LN_attn_resid = self.transformer_LNs[idx][0](attn_resid)
        LN_attn_resid = nn.LayerNorm(self.hidden_dim).to(self.device)(attn_resid)
        # feed forward layer
        FNN_out = self.FNN(idx, LN_attn_resid)
        if self.dropout:
            FNN_out = nn.Dropout(self.dropout)(FNN_out)
        # the second residual
        #FNN_resid = FNN_out + LN_attn_resid
        transformer_output = FNN_out + attn_resid
        # the second layer normalization
        #transformer_output = self.transformer_LNs[idx][1](FNN_resid)
        #transformer_output = nn.LayerNorm(self.hidden_dim).to(self.device)(FNN_resid)

        return transformer_output

    def GAT_layer_0e(self, input_batch, adj_m, idx, bond_embedding):
        # attention layer in Graph Attention
        # combination of node attention and edge attention
        input_batch = nn.LayerNorm(self.hidden_dim).to(self.device)(input_batch)
        q = self.split_head(self.q_layers[idx](input_batch))
        k = self.split_head(self.k_layers[idx](input_batch))
        v = self.split_head(self.v_layers[idx](input_batch))
        ew = self.ew_layers[idx](bond_embedding)
        attn_out = self.multi_head_attention_layer(q, k, v, idx, self.split_bond_embedding(ew), adj_m)
        if self.dropout:
            attn_out = nn.Dropout(self.dropout)(attn_out)
        # the first residual
        attn_resid = attn_out + input_batch
        # the first layer normalization
        LN_attn_resid = nn.LayerNorm(self.hidden_dim).to(self.device)(attn_resid)
        # feed forward layer
        FNN_out = self.FNN(idx, LN_attn_resid)
        if self.dropout:
            FNN_out = nn.Dropout(self.dropout)(FNN_out)
        # the second residual
        transformer_output = FNN_out + attn_resid
        # edge attention
        #print(transformer_output.shape)
        #print(ew.shape)
        edge_output = self.edge_attention(transformer_output, ew)
        return transformer_output, edge_output

    def GAT_layer_1(self, input_batch, adj_m, idx, bond_embedding):
        # attention layer in Graph Attention
        q = self.split_head(self.q_layers[idx](input_batch))
        k = self.split_head(self.k_layers[idx](input_batch))
        v = self.split_head(self.v_layers[idx](input_batch))
        attn_out = self.multi_head_attention_layer(q, k, v, idx, bond_embedding, adj_m)
        if self.dropout:
            attn_out = nn.Dropout(self.dropout)(attn_out)
        # the first residual
        attn_resid = attn_out + input_batch
        # the first layer normalization
        LN_attn_resid = nn.LayerNorm(self.hidden_dim).to(self.device)(attn_resid)
        # feed forward layer
        FNN_out = self.FNN(idx, LN_attn_resid)
        if self.dropout:
            FNN_out = nn.Dropout(self.dropout)(FNN_out)
        # the second residual
        FNN_resid = FNN_out + LN_attn_resid
        # the second layer normalization
        transformer_output = nn.LayerNorm(self.hidden_dim).to(self.device)(FNN_resid)

        return transformer_output

    def get_cls(self, gat_output):
        # get the [CLS] of each mol-graph in the batch
        cls_vector = gat_output[:, 0, :]
        cls_output = self.output_layer(cls_vector)
        cls_output = nn.Dropout(p=0.1)(cls_output)
        cls_output = self.predictor_layer(cls_output)
        if self.prediction_class > 1:
            cls_output = self.predictor_softmax(cls_output)
        #else:
            #cls_output = cls_output.reshape(-1)
        return cls_output

    def forward_(self, input_batch, adj_m):
        # parameter : input_batch ---> Atom information
        # parameter : adj_m ---> Adjacency matrix
        atom_embedding = self.atom_embedding_layer(input_batch)
        if self.bond_influence:
            bond_embedding = self.bond_embedding_layer(adj_m)
            #bond_embedding = self.split_bond_embedding(bond_embedding)
        else:
            bond_embedding = None
        g_layer_output = self.GAT_layer_0(atom_embedding, adj_m, 0, bond_embedding)
        for layer_idx in range(1, self.layer_num):
            g_layer_output = self.GAT_layer_0(g_layer_output, adj_m, layer_idx, bond_embedding)
        prediction = self.get_cls(g_layer_output)
        return prediction
    
    def forward(self, input_batch, adj_m):
        # forward propagation with edge attention
        # parameter : input_batch ---> Atom information
        # parameter : adj_m ---> Adjacency matrix
        atom_embedding = self.atom_embedding_layer(input_batch)
        bond_embedding = self.bond_embedding_layer(adj_m)
        g_layer_output, g_layer_egde = self.GAT_layer_0e(atom_embedding, adj_m, 0, bond_embedding)
        for layer_idx in range(1, self.layer_num):
            g_layer_output, g_layer_egde = self.GAT_layer_0e(g_layer_output, adj_m, layer_idx, g_layer_egde)
        prediction = self.get_cls(g_layer_output)
        return prediction

if __name__ == '__main__':
    hidden_dim = 512
    layer_num = 12
    head_num = 8
    dropout = 0.2
    bond_influence = 1
    prediction_class = 2
    device = 'cuda'
    graph_provider = SMI_grapher(for_predictor=True, device=device)
    graph_provider.fit_new(batch_smis)
    GAT_model = GAT_predictor(hidden_dim, layer_num, head_num, grapher_provider.dict_size, dropout, bond_influence, prediction_class, device)
