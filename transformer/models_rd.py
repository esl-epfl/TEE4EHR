import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
os.add_dll_directory('c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin')
os.add_dll_directory(os.path.dirname(__file__))

from torch.nn.parameter import Parameter
import torch_geometric.nn.inits
from torch_geometric.nn.inits import uniform, glorot, zeros, ones, reset


from transformer.transformer_conv import TransformerConv
from transformer.Ob_propagation import Observation_progation
import warnings
import numbers

# ADDED ****************************
import transformer.Constants as Constants
def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out
# *************************************************



class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        times = torch.Tensor(P_time.cpu()).unsqueeze(2)
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        pe = pe.cuda()
        return pe


class Raindrop_v2(nn.Module):
    """Implement the raindrop stratey one by one."""
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length 
        MAX  = positional encoder MAX parameter
        n_classes = number of classes 
    """

    def __init__(self, d_inp=36, d_model=64, nhead=4, nhid=128, nlayers=2, dropout=0.3, max_len=215, d_static=9,
                 MAX=100, perc=0.5, aggreg='mean', n_classes=2, global_structure=None, sensor_wise_mask=False, static=False):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        self.global_structure = global_structure
        self.sensor_wise_mask = sensor_wise_mask

        d_pe = 16
        d_enc = d_inp

        self.d_inp = d_inp
        self.d_model = d_model
        self.static = static
        if self.static:
            self.emb = nn.Linear(d_static, d_inp)

        self.d_ob = int(d_model/d_inp)

        self.encoder = nn.Linear(d_inp*self.d_ob, self.d_inp*self.d_ob)

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        if self.sensor_wise_mask == True:
            encoder_layers = TransformerEncoderLayer(self.d_inp*(self.d_ob+16), nhead, nhid, dropout)
        else:
            encoder_layers = TransformerEncoderLayer(d_model+16, nhead, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.adj = torch.ones([self.d_inp, self.d_inp]).cuda()

        self.R_u = Parameter(torch.Tensor(1, self.d_inp*self.d_ob)).cuda()

        self.ob_propagation = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                    n_nodes=d_inp, ob_dim=self.d_ob)



        self.ob_propagation_layer2 = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                           n_nodes=d_inp, ob_dim=self.d_ob)

        if static == False:
            d_final = d_model + d_pe
        else:
            d_final = d_model + d_pe + d_inp

        # self.mlp_static = nn.Sequential(
        #     nn.Linear(d_final, d_final),
        #     nn.ReLU(),
        #     nn.Linear(d_final, n_classes),
        # )

        # self.mlp = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, n_classes),
        # )


        # ***************************************************
        self.num_types = d_inp

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model+d_pe, d_inp)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # prediction of next time stamp
        self.time_predictor = Predictor(d_model+d_pe, 1)

        # prediction of next event type
        self.type_predictor = Predictor(d_model+d_pe, d_inp)
        # ***************************************************

        self.aggreg = aggreg
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)
        glorot(self.R_u)

    def forward(self, event_type, event_time, static=None):
        """Input to the model:
        src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector [L, B, 2*d_inp(temporal)]
        static = Pstatic: [128, 9]: this one doesn't matter; static features [B, d_static]
        times = Ptime: [215, 128]: the timestamps [L,B]
        lengths = lengths: [128]: the number of nonzero recordings. [B]
        """
        # MODIFICATIONS
        # event_type, event_time --> src, static, times, lengths
        # event_type, event_time --> src[L,B,2*d_inp], times[L,B], lengths(int[B])

        non_pad_mask = get_non_pad_mask(event_type) # [B,L,1] 0 for padded elements

        times = torch.transpose(event_time,0,1)
        lengths = non_pad_mask.squeeze(-1).sum(-1).type(torch.int)
        
        src = torch.max(event_type-1,torch.tensor([0.]).to('cuda')).type(torch.int64)
        src = nn.functional.one_hot(src, num_classes=self.d_inp)
        src = torch.transpose(src,0,1)
        src = torch.cat([src,src],dim=-1) # [L,B,2*d_inp]


        maxlen, batch_size = src.shape[0], src.shape[1]
        missing_mask = src[:, :, self.d_inp:int(2*self.d_inp)]
        src = src[:, :, :int(src.shape[2]/2)]
        n_sensor = self.d_inp

        src = torch.repeat_interleave(src, self.d_ob, dim=-1)
        h = F.relu(src*self.R_u)
        pe = self.pos_encoder(times)
        # if static is not None:
        #     emb = self.emb(static)

        h = self.dropout(h)

        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).cuda() # [L,B]

        step1 = True
        x = h
        if step1 == False:
            output = x
            distance = 0
        elif step1 == True:
            adj = self.global_structure.cuda() # [d_inp, d_inp]
            adj[torch.eye(self.d_inp).bool()] = 1

            edge_index = torch.nonzero(adj).T # [2, d_inp**2]
            edge_weights = adj[edge_index[0], edge_index[1]]

            batch_size = src.shape[1]
            n_step = src.shape[0]
            output = torch.zeros([n_step, batch_size, self.d_inp*self.d_ob]).cuda() # [L, B, ]

            use_beta = True
            if use_beta == True:
                alpha_all = torch.zeros([int(edge_index.shape[1]/2), batch_size]).cuda() # [1/2 * d_inp**2, B]
            else:
                alpha_all = torch.zeros([edge_index.shape[1],  batch_size]).cuda()
            for unit in range(0, batch_size):
                stepdata = x[:, unit, :]
                p_t = pe[:, unit, :]

                stepdata = stepdata.reshape([n_step, self.d_inp, self.d_ob]).permute(1, 0, 2)
                stepdata = stepdata.reshape(self.d_inp, n_step*self.d_ob)

                stepdata, attentionweights = self.ob_propagation(stepdata, p_t=p_t, edge_index=edge_index, edge_weights=edge_weights,
                                 use_beta=use_beta,  edge_attr=None, return_attention_weights=True)

                edge_index_layer2 = attentionweights[0]
                edge_weights_layer2 = attentionweights[1].squeeze(-1)

                stepdata, attentionweights = self.ob_propagation_layer2(stepdata, p_t=p_t, edge_index=edge_index_layer2, edge_weights=edge_weights_layer2,
                                 use_beta=False,  edge_attr=None, return_attention_weights=True)

                stepdata = stepdata.view([self.d_inp, n_step, self.d_ob])
                stepdata = stepdata.permute([1, 0, 2])
                stepdata = stepdata.reshape([-1, self.d_inp*self.d_ob])

                output[:, unit, :] = stepdata
                alpha_all[:, unit] = attentionweights[1].squeeze(-1)    # [d_inp**2, B]

            distance = torch.cdist(alpha_all.T, alpha_all.T, p=2) # [B,B]
            distance = torch.mean(distance) # []

        if self.sensor_wise_mask == True:
            extend_output = output.view(-1, batch_size, self.d_inp, self.d_ob)
            extended_pe = pe.unsqueeze(2).repeat([1, 1, self.d_inp, 1])
            output = torch.cat([extend_output, extended_pe], dim=-1)
            output = output.view(-1, batch_size, self.d_inp*(self.d_ob+16))
        else:
            output = torch.cat([output, pe], axis=2)

        step2 = True
        if step2 == True:
            r_out = self.transformer_encoder(output, src_key_padding_mask=mask)
        elif step2 == False:
            r_out = output

        sensor_wise_mask = self.sensor_wise_mask

        masked_agg = True
        if masked_agg == True:
            lengths2 = lengths.unsqueeze(1) #[B,1]
            mask2 = mask.permute(1, 0).unsqueeze(2).long() # [L,B,1]
            if sensor_wise_mask:
                output = torch.zeros([batch_size,self.d_inp, self.d_ob+16]).cuda()
                extended_missing_mask = missing_mask.view(-1, batch_size, self.d_inp)
                for se in range(self.d_inp):
                    r_out = r_out.view(-1, batch_size, self.d_inp, (self.d_ob+16))
                    out = r_out[:, :, se, :]
                    len = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)
                    out_sensor = torch.sum(out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (len + 1)
                    output[:, se, :] = out_sensor
                output = output.view([-1, self.d_inp*(self.d_ob+16)])
            elif self.aggreg == 'mean':
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        elif masked_agg == False:
            output = r_out[-1, :, :].squeeze(0)

        if static is not None:
            emb = self.emb(static)
            output = torch.cat([output, emb], dim=1)
        # output = self.mlp_static(output)


        # MODIFICATIONS **********************************************************
        output = torch.transpose(r_out,0,1) # [B,L,d_model+d_pe]

        # [B,L,1] <- [B,L,d_model+d_pe]
        time_prediction = self.time_predictor(output, non_pad_mask)
        
        # [B,L,C] <- [B,L,d_model+d_pe]
        type_prediction = self.type_predictor(output, non_pad_mask)


        # return output, distance, None
        return output, (type_prediction, time_prediction, distance)
