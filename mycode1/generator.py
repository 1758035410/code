import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from model_utils import *


class Generator(nn.Module):
    def __init__(self, args,SE,  window_size=3, T=12, N=None):
        super(Generator, self).__init__()
        K = args.K
        d = args.d
        self.num_his = args.num_his
        self.SE = SE.to(device)
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.AGGCN = AGGCN(args)
        self.ALMA = ALMA(K, d, T=T, window_size=window_size, N=N,SE=self.SE)
        self.end_conv = nn.Conv2d(1, args.num_pred * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.layernorm = nn.LayerNorm(self.hidden_dim, eps=1e-12)
        self.out_dropout = nn.Dropout(0.1)
        self.linear_transform = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.linear_gate = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, TE):
        X = torch.unsqueeze(X, -1)
        aggcn_output = self.AGGCN(X)
        alma_output = self.ALMA(X,TE)
        combined_output = torch.cat((alma_output, aggcn_output), dim=-1)
        attention_weights = self.sigmoid(self.linear_transform(combined_output))
        gate_vector = self.sigmoid(self.linear_gate(combined_output))
        weighted_output = attention_weights * gate_vector * alma_output + (1 - attention_weights * gate_vector) * aggcn_output
        output = self.out_dropout(self.layernorm(weighted_output[:, -1:, :, :]))
        output = self.end_conv((output))


        return torch.squeeze(output, 3)



class ALMA(nn.Module):
    def __init__(self, K, d, T=12, window_size=5, N=None,SE=None):
        super(ALMA, self).__init__()
        self.SE = SE
        self.ALMAttention = ALMAttention(K, d, T=T, window_size=window_size, N=N)
        self.mlp = CONVs(input_dims=[1, K * d], units=[K * d, K * d], activations=[F.relu, None])
        self.PostProcess = PostProcess(K * d)
        self.STEmbedding = STEmbedding(K * d, emb_dim=SE.shape[1]).to(device)
    def forward(self, X,TE):
        X = self.mlp(X)
        STE = self.STEmbedding(self.SE, TE)
        STE_his = STE[:, :12]
        for i in range(2):
            alma_output = self.ALMAttention(X, STE_his)
            alma_output= self.PostProcess(alma_output)
        return alma_output

class ALMAttention(nn.Module):
    def __init__(self, K, d, T=12, window_size=5, N=None):
        super(ALMAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.window = window_size
        self.T = T
        self.N = N

        self.dropout = 0.1
        self.FC_q = nn.Linear(2 * D, D)
        self.FC_k = nn.Linear(2 * D, D)
        self.FC_v = nn.Linear(2 * D, D)

        self.nodevec1 = nn.Parameter(torch.randn(N, 20).cuda(), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(20, N).cuda(), requires_grad=True)

        self.output = TransformerSelfOutput(D, D)
        self.shift_list = self.get_shift_list()

    def get_shift_list(self):
        idxs = np.arange(self.T)
        window_size = self.window
        window_list = np.arange(-(window_size - 1) // 2, (window_size - 1) // 2 + 1, 1)
        shift_list = []
        for i in window_list:
            tmp = idxs + i
            tmp[tmp < 0] = tmp[tmp < 0] + window_size
            tmp[tmp > (self.T - 1)] = tmp[tmp > (self.T - 1)] - window_size
            shift_list.append(tmp)
        shift_list = np.array(shift_list)
        return shift_list

    def get_adp_graph(self, max_num_neigh=40):
        adp_A = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        threshold = 1 / self.N
        tmp, _ = torch.kthvalue(-1 * adp_A, max_num_neigh + 1, dim=1, keepdim=True)
        bin_mask = (torch.logical_and((adp_A > threshold), (adp_A > -tmp)).type_as(adp_A) - adp_A).detach() + adp_A
        adp_A = adp_A * bin_mask
        idxs = torch.nonzero(adp_A)
        src, dst = idxs[:, 0], idxs[:, 1]
        adp_g = dgl.graph((src, dst)).to("cuda")
        return adp_g

    def forward(self, X, STE):
        X_STE = torch.cat((X, STE), dim=-1)
        query = self.FC_q(X_STE)
        key = self.FC_k(X_STE)
        value = self.FC_v(X_STE)
        B = query.shape[0]
        T = query.shape[1]
        N = query.shape[2]
        hdim = query.shape[3] // self.K
        query = query.view(B, T, N, self.K, hdim)
        key = key.view(B, T, N, self.K, hdim)
        value = value.view(B, T, N, self.K, hdim)
        query = query.permute(2, 0, 1, 3, 4)
        value = value.permute(2, 0, 1, 3, 4)
        key = key.permute(2, 0, 1, 3, 4)

        g = self.get_adp_graph()
        g = g.local_var()
        res = 0
        for ti in range(len(self.shift_list)):
            g.ndata['q'] = query / (hdim ** 0.5)
            g.ndata['k'] = key[:, :, self.shift_list[ti], :, :]
            g.ndata['v'] = value[:, :, self.shift_list[ti], :, :]
            g.apply_edges(fn.u_dot_v('k', 'q', 'score'))
            # g.apply_edges(mask_attention_score)
            e = g.edata.pop('score')
            g.edata['score'] = edge_softmax(g, e)
            g.edata['score'] = nn.functional.dropout(g.edata['score'], p=self.dropout, training=self.training)
            g.update_all(fn.u_mul_e('v', 'score', 'm'), fn.sum('m', 'h'))
            output = g.ndata['h']
            output = output.permute(1, 2, 0, 3, 4)
            output = output.reshape(B, T, N, self.K * hdim)
            res += output
        res /= len(self.shift_list)
        output = self.output(res, X)
        return output



class AGGCN(nn.Module):
    def __init__(self, args):
        super(AGGCN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.window = args.num_his
        self.horizon = args.num_pred
        self.num_layers = args.num_layers
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.time_embeddings = nn.Parameter(torch.randn(self.window, args.embed_dim), requires_grad=True)
        self.encoder = AGRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k, args.embed_dim,
                              args.num_layers)


    def forward(self, source):
        init_state = self.encoder.init_hidden(source.shape[0])
        output= self.encoder(source,init_state, self.node_embeddings, self.time_embeddings)

        return output


class AGRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AGRNN, self).__init__()
        assert num_layers >= 1, 'At least one GRU layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dagrnn_cells = nn.ModuleList()
        self.dagrnn_cells.append(GRU(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dagrnn_cells.append(GRU(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings, time_embeddings):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x


        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dagrnn_cells[i](current_inputs[:, t, :, :],  state, node_embeddings,time_embeddings[t])
                inner_states.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dagrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)


class GRU(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(GRU, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = GCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim, node_num)
        self.update = GCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim, node_num)

    def forward(self, x, state, node_embeddings,  time_embeddings):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state,  node_embeddings,  time_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate,  node_embeddings, time_embeddings))
        h = r * state + (1 - r) * hc

        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, node_num):
        super(GCN, self).__init__()
        self.cheb_k = cheb_k
        self.node_num = node_num
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.embs_dropout = nn.Dropout(0.1)

    def forward(self, x, node_embeddings,  time_embeddings):
        node_embeddings = self.embs_dropout(self.layernorm(node_embeddings + time_embeddings.unsqueeze(0)))
        supports = F.softmax(torch.mm(node_embeddings, node_embeddings.transpose(0, 1)), dim=1)
        support_set = [torch.eye(self.node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)

        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)

        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias

        return x_gconv


