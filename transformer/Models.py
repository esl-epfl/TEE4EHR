import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer

from transformer.Modules import  Predictor, CIF_sahp, MLP_state, CumulativeSetAttentionLayer

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    # assert seq.dim() == 2
    # return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

    if seq.dim() == 2:
        return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)
    else:
        return seq.sum(-1).ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    if seq_k.dim() == 2:

        padding_mask = seq_k.eq(Constants.PAD)
    else:
        padding_mask = seq_k.sum(-1).eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq, diag_offset=1):
    """ For masking out the subsequent info, i.e., masked self-attention. """
    if seq.dim() == 2:
        sz_b, len_s = seq.size()
    else:
        sz_b, len_s, dim = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=diag_offset)
    subsequent_mask = subsequent_mask.unsqueeze(
        0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


def softmax_weighting(values, preattention, mask, eps=1e-7, online=True):
    """Cumulative softmax weighting of values.

    Args:
        values: Values expected shape [n_samples, feature_dim] [B,P,d_phi]
        preattention: Preattention values, expected shape [n_samples, n_heads] [B,P,m]
        segment_ids: Segment ids

    Returns:
    """

    # out5=[]
    att = nn.Softmax(dim=1)(preattention)  # [B,P,m]
    out5 = torch.einsum('BPm,BPd->Bmd', att, values)  # [B,m,d]
    # out5 = out5 * mask.unsqueeze(-1)

    if out5.isnan().sum():
        # torch.nan_to_num(out5, nan=0)
        aa = 1
        print('###################### Nan values !!!!!!!!!!!!!!!')
    # else:
    #     print("###")

    # for cur_head_preattn in head_preattn:
    #     # For numerical stability subtract the max from data values
    #     max_values = torch.math.segment_max(cur_head_preattn, segment_ids)
    #     max_values = torch.gather_nd(max_values, torch.expand_dims(segment_ids, -1))
    #     max_values = torch.stop_gradient(max_values)

    #     normalized = cur_head_preattn - max_values
    #     exp_preattn = torch.exp(normalized)
    #     exp_head_preattn.append(exp_preattn)
    #     cumulative_exp_preattn.append(
    #         cumulative_segment_sum(
    #             exp_preattn, segment_ids))

    # exp_head_preattn = torch.stack(exp_head_preattn, -1)
    # weighted_values = \
    #     torch.expand_dims(values, 1) * torch.expand_dims(exp_head_preattn, -1)

    # cumulative_exp_preattn = torch.stack(cumulative_exp_preattn, axis=-1)

    # # Sum the values
    # out = (
    #     (cumulative_segment_sum(weighted_values, segment_ids) + eps)
    #     / (torch.expand_dims(cumulative_exp_preattn, -1) + eps)
    # )
    return out5, att


def cumulative_softmax_weighting(values, preattention, mask, eps=1e-7, online=True):
    """Cumulative softmax weighting of values.

    Args:
        values: Values expected shape [n_samples, feature_dim] [B,P,d_phi]
        preattention: Preattention values, expected shape [n_samples, n_heads] [B,P,m]
        segment_ids: Segment ids

    Returns:
    """
    # cumulative_segment_sum = cumulative_segment_wrapper(torch.cumsum)

    # head_preattn = torch.unstack(preattention, axis=-1)
    # exp_head_preattn = []
    # cumulative_exp_preattn = []

    # max_preattention = torch.cummax(preattention, dim=1)[0] # [B,P,m]
    # exp_preattn = torch.exp(preattention-max_preattention).unsqueeze(-1) # [B, P,m, 1]
    # cumsum_exp_preattn = torch.cumsum(exp_preattn, dim=1) # [B, P,m, 1] across first dim
    # out1 = exp_preattn @ values.unsqueeze(-2) / cumsum_exp_preattn # [B,P,m,d_phi]

    # max_preattention2 = torch.max(preattention, dim=1)[0].unsqueeze(1) # [B,P,m]
    # exp_preattn2 = torch.exp(preattention-max_preattention2).unsqueeze(-1) # [B, P,m, 1]
    # cumsum_exp_preattn2 = torch.cumsum(exp_preattn2, dim=1) # [B, P,m, 1] across first dim
    # out2 = exp_preattn2 @ values.unsqueeze(-2) / cumsum_exp_preattn2 # [B,P,m,d_phi]

    # exp_preattn3 = torch.exp(preattention).unsqueeze(-1) # [B, P,m, 1]
    # cumsum_exp_preattn3 = torch.cumsum(exp_preattn3, dim=1) # [B, P,m, 1] across first dim
    # out3 = exp_preattn3 @ values.unsqueeze(-2) / cumsum_exp_preattn3 # [B,P,m,d_phi]

    # out4 = []

    # for i in range(preattention.shape[1]):

    #     max_preattention4 = torch.max(preattention[:,:i+1,:], dim=1)[0].unsqueeze(1).detach() # [B,1,m]
    #     exp_preattn4 = torch.exp(preattention[:,:i+1,:]-max_preattention4).unsqueeze(-1) # [B, i+1,m, 1]
    #     cumsum_exp_preattn4 = torch.cumsum(exp_preattn4, dim=1) # [B, i+1,m, 1] across first dim
    #     temp = exp_preattn4 @ values[:,:i+1,:].unsqueeze(-2) / cumsum_exp_preattn4 # [B,i+1,m,d_phi]
    #     temp = (exp_preattn4 @ values[:,:i+1,:].unsqueeze(-2)).sum(1) / cumsum_exp_preattn4[:,-1,:,:] # [B,i+1,m,d_phi]

    #     # temp = nn.Softmax(dim=1)(preattention[:,:i+1,:]).unsqueeze(-1) @ values[:,:i+1,:].unsqueeze(-2)

    #     out4.append(temp[:,[-1],:,:])
    # out4 = torch.concat(out4, dim=1)

    # out4 = out4 * mask.unsqueeze(-1)

    # out5=[]
    A = preattention[:, None, :, :]  # [B,1,P,m]
    B = torch.cummax(preattention, dim=1)[0][:, :, None, :]  # [B,P,1,m]
    C = torch.triu(
        torch.ones(
            (preattention.shape[1], preattention.shape[1]), device=preattention.device).bool(),
        diagonal=1)[None, :, :, None]  # [1,P,P,1]

    D = torch.exp((A-B).masked_fill(C, -np.inf))  # [B,P,P,m]
    E = D.sum(2)[:, :, :, None]  # [B,P,m,1]
    # out5 = torch.exp(D)[:,:,:,:,None] @ values[:,:,None,None,:] / E
    # out5 = torch.einsum('BPPm1,BP11d->BP1md',torch.exp(D)[:,:,:,:,None] , values[:,:,None,None,:])
    out5 = torch.einsum('BPQm,BQd->BPmd', D, values) / (E)  # [B,P,m,d]
    out5 = out5 * mask.unsqueeze(-1)

    # if online:
    #     # att = torch.einsum('BPQm,BPm->BPQm',D , 1/E.squeeze())  # [B,P,m,d]

    #     att = (D / E[:,:,None,:,0])   # [B,P,P,m]
    # else:
    #     att = (D / E[:,:,None,:,0])   # [B,P,P,m]

    att = []

    # new way: more stable
    # sm = nn.Softmax(dim=1)
    # out = []
    # for i in range(preattention.shape[1]): # iterate over P

    #     v1 = torch.permute( sm(preattention[:,:i+1,:].unsqueeze(-1)) , (0,2,3,1)) # [B,*,m,,1] -> [B,m,1,*]
    #     v2 = values[:,:i+1,:].unsqueeze(1) # [B,*,d]->[B,1,*,d]
    #     temp = torch.matmul(v1,v2).squeeze(-2) # [B,m,1,d] -> [B,m,d]
    #     # [B,*,m,1] @ [B,*,1,d_phi]

    #     out.append(temp)
    # out = torch.stack(out,dim=1) # [B,P,m,d]
    # out = out * mask.unsqueeze(-1)

    if out5.isnan().sum():
        # torch.nan_to_num(out5, nan=0)
        aa = 1
        print('###################### Nan values !!!!!!!!!!!!!!!')
    # else:
    #     print("###")

    # for cur_head_preattn in head_preattn:
    #     # For numerical stability subtract the max from data values
    #     max_values = torch.math.segment_max(cur_head_preattn, segment_ids)
    #     max_values = torch.gather_nd(max_values, torch.expand_dims(segment_ids, -1))
    #     max_values = torch.stop_gradient(max_values)

    #     normalized = cur_head_preattn - max_values
    #     exp_preattn = torch.exp(normalized)
    #     exp_head_preattn.append(exp_preattn)
    #     cumulative_exp_preattn.append(
    #         cumulative_segment_sum(
    #             exp_preattn, segment_ids))

    # exp_head_preattn = torch.stack(exp_head_preattn, -1)
    # weighted_values = \
    #     torch.expand_dims(values, 1) * torch.expand_dims(exp_head_preattn, -1)

    # cumulative_exp_preattn = torch.stack(cumulative_exp_preattn, axis=-1)

    # # Sum the values
    # out = (
    #     (cumulative_segment_sum(weighted_values, segment_ids) + eps)
    #     / (torch.expand_dims(cumulative_exp_preattn, -1) + eps)
    # )
    return out5, att

def align(r_enc, event_time, state_time):
    """
    event_time [B,L]
    state_time [B,P]
    r_enc [B,P, d]
    """

    diff = event_time.unsqueeze(-1) - state_time.unsqueeze(-2)  # [B,L,P]
    diff[diff <= 0] = 1e9
    # indices = torch.argmax(diff,-1).flatten()-1 # [B*L]
    indices = torch.argmax(diff, -1)-1  # [B,L]
    indices[indices < 0] = 0

    # indices_1hot = F.one_hot(indices,num_classes=r_enc.shape[1]).float() # [B,L,P]
    # # [B,L,d,1,1] = [B,L,1,1,P] @ [B,1,d,P,1]
    # r_enc_red = torch.matmul(indices_1hot[:,:,None,None,:]   ,   r_enc.transpose(1,2)[:,None,:,:,None])
    # r_enc_red = r_enc_red.squeeze(-1).squeeze(-1)  # [B,L,d]

    # r_enc_red2 = torch.zeros(*event_time.shape, r_enc.shape[-1],device=r_enc.device) # [B,L,d]
    # for i,x in enumerate(indices):
    #     r_enc_red2[i]=r_enc[i,indices[i],:]

    r_enc_red = torch.gather(
        r_enc, 1, indices.unsqueeze(-1).repeat(1, 1, r_enc.shape[-1]))

    # this was wrong
    # r_enc_red = torch.index_select(r_enc.reshape(-1, r_enc.shape[-1]), 0, indices.flatten()).reshape(*event_time.shape,-1) # [B,L, d]

    # example
    # print(event_time[1][:5])
    # print(state_time[1][:55])
    # print(r_enc[1,:55])
    # print(r_enc_red[1,:5])
    # print(indices[1,:5])
    # print(r_enc[1,indices[1,:5]])

    # diff[0][:5,:5]
    # torch.argmax(diff,-1)[0,:5]-1

    # r_enc.sum(-1)

    return r_enc_red  # [B,L,d]


class TEE(nn.Module):
    """ A Transformer Event Encoder (TEE) model with self attention mechanism. """

    def __init__(
            self,
            n_marks,
            d_type_emb,

            time_enc,
            d_time,

            d_inner,
            n_layers,
            n_head,
            d_k,
            d_v,
            dropout,

            device,

            diag_offset,
            # reg=False,


    ):
        super().__init__()

        self.diag_offset = diag_offset
        self.n_marks = n_marks
        self.d_type_emb = d_type_emb

        self.time_enc = time_enc
        self.d_time = d_time  # will be set to 0 later if 'sum'

        self.d_inner = d_inner
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout

        # position vector, used for temporal encoding
        if self.time_enc == 'sum':
            self.position_vec = torch.tensor(
                [math.pow(10000.0, 2.0 * (i // 2) / (self.d_type_emb))
                 for i in range(self.d_type_emb)],
                device=device)
            self.d_time = 0

        elif self.time_enc == 'concat':

            self.position_vec = torch.tensor(
                [math.pow(10000.0, 2.0 * (i // 2) / (self.d_time))
                 for i in range(self.d_time)],

                device=device)
        elif self.time_enc == 'none':
            self.d_time = 0

        self.event_emb = nn.Linear(n_marks, d_type_emb, bias=True)
        # nn.init.xavier_uniform_(self.event_emb.weight)

        self.d_model = self.d_type_emb + self.d_time

        # if self.reg:
        #     self.A_reg = torch.nn.Parameter(torch.ones(num_types, num_types))

        # TE_layer = nn.TransformerEncoderLayer(self.d_model, n_head, dim_feedforward=d_inner, dropout=dropout, layer_norm_eps=1e-05, batch_first=True, norm_first=False, device=None, dtype=None)

        # self.layer_stack = nn.ModuleList([
        #     TE_layer
        #     for _ in range(n_layers)])

        self.layer_stack = nn.ModuleList([
            EncoderLayer(self.d_model, d_inner, n_head, d_k, d_v,
                         dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

        # self.layer_stack = nn.ModuleList([nn.TransformerEncoderLayer(self.d_model, n_head, dim_feedforward=d_inner, dropout=dropout, activation=nn.ReLU(), batch_first=True, norm_first=False, device=None, dtype=None)
        #                                   for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_mark.
        """

        # temp = torch.ones_like(time) * torch.arange(time.shape[1],device=time.device) # [B,L] * [L]=[B,L]
        # result = temp.unsqueeze(-1) / self.position_vec

        result = time.unsqueeze(-1) / self.position_vec

        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(
            event_type, self.diag_offset)
        slf_attn_mask_keypad = get_attn_key_pad_mask(
            seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(
            slf_attn_mask_subseq)
        # [B,L,L] True are masked
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        # if len(event_type.shape)==3:
        #     # E_oh = nn.functional.one_hot(event_type.sum(-1).bool().long(), num_classes=self.num_types+1)[:,:,1:].type(torch.float)    # [B,L,num_classes]
        #     E_oh = event_type
        # else:
        #     E_oh = nn.functional.one_hot(event_type, num_classes=self.num_types+1)[:,:,1:].type(torch.float)

        # [B,L,L] <- [B,L,num_classes] * [num_classes,num_classes] * [B,num_classes,L]
        # mask2 = torch.matmul( torch.matmul(E_oh,A_reg) , E_oh.transpose(1,2))   # 1 INDICATES a connection
        # # mask2 = mask2 * (1 - torch.triu(mask2, diagonal=1))
        # mask2 = (torch.triu(mask2, diagonal=1))

        # slf_attn_mask = slf_attn_mask #+ (mask2) # [B,L,L] True are masked (do not attend)

        # ### Event type embedding
        if len(event_type.shape) == 3:
            # E_oh = nn.functional.one_hot(event_type.sum(-1).bool().long(), num_classes=self.num_types+1)[:,:,1:].type(torch.float)    # [B,L,num_classes]
            event_type_1hot = event_type.type(torch.float)
        else:
            event_type_1hot = nn.functional.one_hot(
                event_type, num_classes=self.n_marks+1)[:, :, 1:].type(torch.float)

        x = self.event_emb(event_type_1hot)

        # ### Event time encoding
        time_enc = self.temporal_enc(
            event_time, non_pad_mask)  # [B, L, d_model]

        # ### combining
        if self.time_enc == 'sum':
            x += time_enc  # [B,L,d_model]
        elif self.time_enc == 'concat':
            x = torch.cat((x, time_enc), -1)  # [B,L,d_model]
        elif self.time_enc == 'none':
            pass

        for enc_layer in self.layer_stack:
            # x += tem_enc

            x, self_attn = enc_layer(
                x,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

            # x = enc_layer(
            #     x,
            #     # [L,L] True means: do not attend
            #     src_mask=slf_attn_mask_subseq[0].bool(),
            #     src_key_padding_mask=(~non_pad_mask.bool()).squeeze(-1))  # [b,L] True means ignoring

        # self.self_attn = self_attn
        # self.slf_attn_mask = slf_attn_mask
        # self.mask2 = self_attn#mask2*0

        self.self_attn = self_attn.detach().cpu()

        return x


class DeepAttensionModule(nn.Module):
    def __init__(self, output_activation='relu', output_dims=4,
                 n_phi_layers=2, phi_width=4, phi_dropout=0.1,
                 n_psi_layers=2, psi_width=4, psi_latent_width=8,
                 dot_prod_dim=4, n_heads=2, attn_dropout=0.1, latent_width=8,
                 n_rho_layers=2, rho_width=4, rho_dropout=0.1,
                 max_timescale=10000, d_time=4,
                 num_mods=22,
                 num_demos=0,

                 online=False,
                 device=torch.device('cpu')

                 ):

        super(DeepAttensionModule, self).__init__()

        self.online = online

        self.num_mods = num_mods

        dim_s = num_mods+d_time+1

        self._config = {
            name: val for name, val in locals().items()
            if name not in ['self', '__class__']
        }

        self.return_sequences = True

        self.phi_width = phi_width
        self.rho_width = rho_width

        # self.to_segments = PaddedToSegments()

        # If we set n_positional_dims to 0, skip the positional encoding
        # self.positional_encoding = (
        #     PositionalEncoding(max_timescale, n_positional_dims)
        #     if n_positional_dims != 0
        #     else nn.Identity()
        # )

        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_time)
             for i in range(d_time)],
            device=device)

        # We need the input dimensionality in order to determine the size of
        # the embedding for the demographics.
        # phi_input_dim = transformed_times[-1] + values[-1] + mod_shape

        self.demo_encoder = None
        if num_demos > 0:
            self.demo_encoder = Predictor(num_demos, dim_s)

        # if isinstance(output_dims, Sequence):
        #     # We have an online prediction scenario
        #     assert output_dims[0] is None
        #     self.return_sequences = True
        #     output_dims = output_dims[1]
        # else:
        #     self.return_sequences = False

        # MLP encoder for combined values
        # Build phi architecture
        self.phi = MLP_state(dim_s, phi_width, n_phi_layers, phi_dropout)

        # self.phi.add(Dense(latent_width, **self.dense_options))
        self.latent_width = latent_width
        self.n_heads = n_heads

        self.attention = CumulativeSetAttentionLayer(
            dim_s,
            n_psi_layers, psi_width, psi_latent_width,
            dot_prod_dim=dot_prod_dim, n_heads=n_heads,
            attn_dropout=attn_dropout
        )

        # # Build rho architecture
        self.rho = MLP_state(n_heads*phi_width, rho_width,
                             n_rho_layers, rho_dropout)
        # self.rho = build_dense_dropout_model(
        #     n_rho_layers, rho_width, rho_dropout, self.dense_options)
        # self.rho.add(Dense(output_dims, activation=output_activation))
        # self._n_modalities = None

        self.temp = {}

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_mark.
        """

        result = time.unsqueeze(-1) / self.position_vec[None, None, :]
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result  # * non_pad_mask

    def forward(self,
                # state_time, state_val, state_mark,
                state_data,
                non_pad_mask,
                verbose=False,
                demo=None

                ):

        # demo, times, values, measurements, lengths, elem_per_tp, pred_lengths = inputs
        times = state_data[0]
        values = state_data[1].unsqueeze(-1)
        measurements = state_mark = state_data[2]
        demo = state_data[-1] if len(state_data) == 4 else None
        lengths = non_pad_mask.squeeze(-1).sum(-1).int().tolist()
        pred_lengths = 1

        # if len(pred_lengths.get_shape()) == 2:
        #     pred_lengths = torch.squeeze(pred_lengths, -1)

        # [B, P, d_t] d_t:n_positional_dims
        transformed_times = self.temporal_enc(times, non_pad_mask)

        # Transform modalities
        transformed_measurements = nn.functional.one_hot(
            measurements, num_classes=self.num_mods+1)[:, :, 1:].type(torch.float)    # [B,L,num_mods]

        combined_values = torch.concat(
            (
                transformed_times,
                values,
                transformed_measurements
            ),
            axis=-1
        )  # [B,P, d_s=d_t+1+n_mod]

        # if demo is not None:
        #     demo_encoded = self.demo_encoder(demo,1)[:,None,:] # [B,1, d_s=d_t+1+n_mod]
        #     combined_values = torch.concat(
        #         [demo_encoded, combined_values], axis=1) # [B,P+1, d_s=d_t+1+n_mod]
        #     # combined_with_demo = combined_values

        # Somehow eager execution and graph mode behave differently.
        # In graph mode lengths has an additional dimension
        # if len(lengths.get_shape()) == 2:
        #     lengths = torch.squeeze(lengths, -1)

        # We additionally have the encoded demographics as a set element
        # mask = torch.sequence_mask(lengths+1, name='mask')

        # mask = get_attn_key_pad_mask(seq_k=state_mark, seq_q=state_mark) # [B,P,P]

        # collected_values, segment_ids = self.to_segments(
        #     combined_with_demo, mask)

        # preattentions = self.attention(collected_values, segment_ids) # E matrix [B,P,m]

        preattentions = self.attention(
            combined_values, non_pad_mask)  # E matrix [B,P,m]

        # H matrix [B, P, d_phi]
        encoded = self.phi(combined_values, non_pad_mask)

        # R matrix [P,m,d]
        if self.online:
            agg, att = cumulative_softmax_weighting(
                encoded, preattentions, non_pad_mask)  # [B,P,m,d_phi]
            agg = torch.reshape(
                agg, [agg.shape[0], agg.shape[1], -1])  # [B,P,m*d_phi]

            output = self.rho(agg, non_pad_mask)

        else:
            agg, att = softmax_weighting(
                encoded, preattentions, non_pad_mask)  # [B,m,d_phi]
            agg = torch.reshape(agg, [agg.shape[0], -1])  # [B,m*d_phi]

        # Remove heads dimension

            output = self.rho(agg, 1)
        # output=encoded
        self.att = att.detach().cpu()
        if verbose:
            self.temp['preattentions'] = preattentions
            self.temp['times'] = times
            self.temp['values'] = values
            self.temp['measurements'] = measurements
            self.temp['att'] = att
            self.temp['lengths'] = lengths

        # predictions_mask = torch.sequence_mask(pred_lengths)
        # gathered_time_indices, batch_indices = self.to_segments(
        #     elem_per_tp, predictions_mask)

        # # Compute index of the last observation associated with the
        # # provided time.
        # prediction_indices = torch.math.cumsum(gathered_time_indices)
        # # Add an offset for each instance to account for demographics. This
        # # offset decreases for each later index in the batch. Thus we can
        # # use the batch indices.
        # prediction_indices += batch_indices

        # gathered_embeddings = torch.gather_nd(
        #     agg, prediction_indices[:, None])
        # # Lost shape information
        # gathered_embeddings.set_shape([None, None])
        # output = self.rho(gathered_embeddings)

        # valid_predictions = torch.cast(torch.where(predictions_mask), torch.int32)

        # output = torch.scatter_nd(
        #     valid_predictions,
        #     output,
        #     torch.concat(
        #         [torch.shape(predictions_mask), torch.shape(output)[-1:]],
        #         axis=0
        #     )
        # )
        # # torch.print(torch.shape(output), torch.shape(mask))
        # output._keras_mask = predictions_mask
        return output  # * non_pad_mask


class TEEDAM(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,

            n_marks,

            TE_config=False,
            DAM_config=False,
            NOISE_config=False,

            CIF_config=False,
            next_time_config=False,
            next_type_config=False,
            label_config=False,

            demo_config=False,
            device=torch.device('cpu'),
            diag_offset=1
    ):
        super().__init__()


        self.d_out_te = 0
        self.d_out_dam = 0
        self.d_out_nexttype = 0

        self.n_marks = n_marks

        self.d_out_te = 0
        self.d_out_dam = 0
        self.d_demo = 0
        self.noise_size = 0
        self.add_noise = False

        self.device = device

        self.temp = {}

        # TRANSFORMER ENCODER ************************************************************
        if TE_config:
            self.TE = TEE(
                # num_types = TE_config['num_types'],

                n_marks=TE_config['n_marks'],
                d_type_emb=TE_config['d_type_emb'],

                time_enc=TE_config['time_enc'],
                d_time=TE_config['d_time'],

                d_inner=TE_config['d_inner'],
                n_layers=TE_config['n_layers'],
                n_head=TE_config['n_head'],
                d_k=TE_config['d_k'],
                d_v=TE_config['d_v'],
                dropout=TE_config['dropout'],

                # reg=False,
                device=self.device,
                diag_offset=diag_offset


            )

            self.d_out_te = self.TE.d_model

        # DeepAttensionModule ************************************************************
        self.d_DA = 0
        if DAM_config:

            print('[info] STATE will be considered')

            n_heads_DA = 2
            psi_width = 4
            self.DAM = DeepAttensionModule(
                output_activation=DAM_config['output_activation'],
                output_dims=DAM_config['output_dims'],

                # MLP encoder for combined values
                n_phi_layers=DAM_config['n_phi_layers'],
                phi_width=DAM_config['phi_width'],
                phi_dropout=DAM_config['phi_dropout'],

                # Cumulative Set Attention Layer
                n_psi_layers=DAM_config['n_psi_layers'],
                psi_width=DAM_config['psi_width'],
                psi_latent_width=DAM_config['psi_latent_width'],

                dot_prod_dim=DAM_config['dot_prod_dim'],
                n_heads=DAM_config['n_heads'],
                attn_dropout=DAM_config['attn_dropout'],
                latent_width=DAM_config['latent_width'],

                n_rho_layers=DAM_config['n_rho_layers'],
                rho_width=DAM_config['rho_width'],
                rho_dropout=DAM_config['rho_dropout'],



                max_timescale=DAM_config['max_timescale'],
                d_time=DAM_config['n_positional_dims'],
                num_mods=DAM_config['num_mods'],
                num_demos=DAM_config['num_demos'],

                online=DAM_config['online'],
                device=self.device,


            )
            self.d_out_dam = self.DAM.phi_width * self.DAM.n_heads
            self.d_out_dam = self.DAM.rho_width

        if demo_config:

            self.demo_encoder = Predictor(
                demo_config['num_demos'], demo_config['d_demo'])
            self.d_demo = demo_config['d_demo']

        if NOISE_config:
            self.add_noise = True
            self.noise_size = NOISE_config['noise_size']

        self.d_con = self.d_out_te + self.d_out_dam + self.d_demo + self.noise_size

        if self.d_con == 0:
            raise Exception('### NO solution! d_con=0')

        # sample final label ************************************************************
        if label_config:
            self.pred_label = Predictor(self.d_con, 1)
            self.sample_detach = label_config['sample_detach']

        # Prediction of next time and type ***************************

        if next_time_config:
            self.pred_next_time = Predictor(self.d_con, 1)

        if next_type_config:
            self.pred_next_type = Predictor(
                self.d_con, next_type_config['n_marks'])
            self.mark_detach = next_type_config['mark_detach']

        # LL params **************************************************

        if CIF_config:

            if CIF_config['type'] == 'sahp':
                self.event_decoder = CIF_sahp(
                    self.d_con, CIF_config['n_cifs'], mod_CIF=CIF_config['mod'])
            elif CIF_config['type'] == 'thp':
                self.event_decoder = CIF_thp(self.d_con, CIF_config['n_cifs'])

    def forward(self,
                # event_type, event_time,
                # state_time=None, state_val=None, state_mark=None,
                # verbose=False,


                event_type, event_time, state_data=None,  # label_data=None,
                verbose=False,
                ):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
               state_data [state_time, state_val, state_mark,(demo)]
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        non_pad_mask = get_non_pad_mask(
            event_type)  # [B,L,1] 0 for padded elements

        enc = []

        if hasattr(self, 'TE'):
            x = self.TE(event_type, event_time, non_pad_mask)
            # x = x + torch.randn_like(x)*0 # [B,L,nosie_size]
            enc.append(x)

        # # [B,L,d_mark] <- [B,L], [B,L], [B,L,1]
        # if self.event_enc:
        #     h_enc = self.encoder(event_type, event_time, non_pad_mask, self.A_reg,verbose=verbose)
        #     enc.append(h_enc)

        # self.A_reg = self.global_structure * self.A_reg

        if hasattr(self, 'DAM'):
            non_pad_mask_state = get_non_pad_mask(state_data[0])  # [B,P,1]
            # r_enc = self.DAM(state_time, state_val, state_mark, non_pad_mask)
            r_enc = self.DAM(state_data, non_pad_mask_state,
                             verbose=verbose)  # [B,P, m*d_phi]

            # temp = r_enc.sum(-1)#+(1-non_pad_mask_state.squeeze(-1))
            # a = (temp!=0).sum()/non_pad_mask_state.sum()
            if len(r_enc.shape) == 3:   # online scenario
                r_enc_red = align(r_enc, event_time,
                                  state_data[0])  # [B,L, m*d_phi]
                r_enc_red = r_enc_red * non_pad_mask
            else:
                r_enc_red = r_enc[:, None, :].repeat(1, event_time.shape[1], 1)
            # temp = r_enc_red.sum(-1)#+(1-non_pad_mask_state.squeeze(-1))
            # b = (temp!=0).sum()/non_pad_mask.sum()

            enc.append(r_enc_red)

        if self.add_noise:
            # enc[0] is TE embeddings or DAM embeddings

            r_noise = torch.randn(
                *list(enc[0].shape[:2]), self.noise_size, device=enc[0].device)  # [B,L,nosie_size]
            r_noise = r_noise * non_pad_mask
            enc.append(r_noise)

            # enc[0] = enc[0] + torch.randn_like(enc[0]) * 0.1

        if hasattr(self, 'demo_encoder'):
            enc.append(self.demo_encoder(
                state_data[-1], 1)[:, None, :].repeat(1, event_time.shape[1], 1))

        enc = torch.cat(enc, dim=-1)

        # Outputs
        if hasattr(self, 'pred_next_time'):
            # [B,L,1] <- [B,L,d_mark]
            self.y_next_time = self.pred_next_time(enc.detach(), non_pad_mask)

        if hasattr(self, 'pred_next_type'):
            # [B,L,C] <- [B,L,d_mark]
            self.y_next_type = self.pred_next_type(
                enc, non_pad_mask, to_detach=self.mark_detach)
            if torch.isnan(self.y_next_type).sum() > 0:
                a = 1

        # if hasattr(self, 'event_decoder'):
        #     log_sum, integral_ = self.event_decoder(enc,event_time, event_time, non_pad_mask)
        #     self.loss.CIF = -torch.sum(log_sum - integral_)

        # state_label_prediction=0
        if hasattr(self, 'pred_label'):

            lens = non_pad_mask.squeeze(-1).sum(-1).long()
            temp = torch.unbind(enc)
            # temp = [temp[lens[i]] for i in range (len(temp))]
            enc_last = torch.stack([temp[i][lens[i].item() - 1]
                                   for i in range(len(temp))], 0)  # [B,d_con]
            # enc_last = enc_last if self.sample_label==1 else enc_last.detach() # if it is equal to 2
            # self.y_label = self.pred_label(enc_last, 1) # [B]
            if self.sample_detach:
                self.y_label = self.pred_label(
                    enc.detach(), non_pad_mask)  # [B, L]
            else:
                self.y_label = self.pred_label(enc, non_pad_mask)  # [B, L]

        if verbose:
            lens = non_pad_mask.squeeze(-1).sum(-1).long()
            temp = torch.unbind(enc)
            # temp = [temp[lens[i]] for i in range (len(temp))]
            enc_last = torch.stack([temp[i][lens[i].item()-1]
                                   for i in range(len(temp))], 0)
            enc_last = enc[np.arange(enc.shape[0]),  lens-1, :]
            self.temp['enc_last'] = enc_last.cpu().numpy()
            # self.temp['samp_pred']=state_label_prediction.gt(0).type(torch.int).squeeze(-1).cpu().numpy() # [B]
            # self.temp['samp_score']=state_label_prediction.cpu().numpy() # [B]

        return enc
