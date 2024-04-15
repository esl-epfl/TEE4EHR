import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer.Constants as Constants

import Utils


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    # assert seq.dim() == 2
    # return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

    if seq.dim() == 2:
        return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)
    else:
        return seq.sum(-1).ne(Constants.PAD).type(torch.float).unsqueeze(-1)



class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -65000)

        attn = self.dropout(F.softmax(attn, dim=-1))
        # if mask is not None:
        #     attn = attn.masked_fill(mask, 0)
        output = torch.matmul(attn, v)

        return output, attn



class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear1 = nn.Linear(dim, 16, bias=True)
        self.linear2 = nn.Linear(16, 16, bias=True)
        self.linear3 = nn.Linear(16, num_types, bias=True)

        self.relu = nn.ReLU()
        # self.do = nn.Dropout(0.2)
        self.do = nn.Dropout(0.0)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.xavier_normal_(self.linear3.weight)

    def forward(self, data, non_pad_mask, to_detach=0):

        if to_detach:
            data = data.detach()
        # out = self.linear1(data* non_pad_mask)* non_pad_mask
        out = self.relu(self.do(self.linear1(data))) * non_pad_mask
        out = self.relu(self.do(self.linear2(out))) * non_pad_mask
        out = ((self.linear3(out))) * non_pad_mask

        # out = self.relu(out)

        # out = self.linear2(out* non_pad_mask)
        # out=nn.Dropout(0.1)(out)

        # out = out * non_pad_mask
        return out


class CIF_sahp(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, d_in, n_cifs, mod_CIF='mc'):
        super().__init__()

        self.d_in = d_in
        self.d_in = d_in
        self.n_cifs = n_cifs

        self.n_mc_samples = 100
        self.mod = mod_CIF

        self.start_layer = nn.Sequential(
            nn.Linear(self.d_in, self.d_in, bias=True),
            GELU()
        )

        self.converge_layer = nn.Sequential(
            nn.Linear(self.d_in, self.d_in, bias=True),
            GELU()
        )

        self.decay_layer = nn.Sequential(
            nn.Linear(self.d_in, self.d_in, bias=True), nn.Softplus(beta=10.0)
        )

        self.intensity_layer = nn.Sequential(
            nn.Linear(self.d_in, self.n_cifs, bias=True),
            nn.Softplus(beta=1.)
        )

        nn.init.xavier_normal_(self.intensity_layer[0].weight)
        nn.init.xavier_normal_(self.start_layer[0].weight)
        nn.init.xavier_normal_(self.converge_layer[0].weight)
        nn.init.xavier_normal_(self.decay_layer[0].weight)

    def state_decay(self, converge_point, start_point, omega, duration_t):
        # * element-wise product
        cell_t = torch.tanh(converge_point + (start_point -
                            converge_point) * torch.exp(- omega * duration_t))

        # cell_t = nn.Softplus()(converge_point + (start_point - converge_point) * torch.exp(- omega * duration_t))

        # cell_t = (converge_point + (staxrt_point - converge_point) * torch.exp(- omega * duration_t))

        return cell_t

    def forward(self, embed_info, seq_times, seq_types, non_pad_mask):

        # event_ll, non_event_ll = opt.event_loss(model, enc_out, event_time, event_type, side = prediction, mod=opt.mod)
        self.n_mc_samples = 100

        n_batch = seq_times.size(0)
        n_times = seq_times.size(1) - 1  # L-1
        device = seq_times.device

        # embed_event = side[-4] # [B,L,d_model]
        # embed_state = side[-2] # [B,L,d_r]
        # embed_event = embed_info[:,:,:self.d_TE] # [B,L,d_model]
        embed_event = embed_info  # [B,L,d_model]

        # embed_state = embed_info[:,:,self.d_TE:] # [B,L,d_r]
        # state_times = side[-1] # [B,P]
        # embed_state = side[-3] # [B,P,d_r]

        # non_pad_mask = get_non_pad_mask(seq_types).squeeze(2)

        if self.mod == 'single':
            # seq_onehot_types=torch.ones_like(seq_times).unsqueeze(-1) # [B,L,1]
            seq_onehot_types = (seq_times > 0).long().unsqueeze(-1)  # [B,L,1]

        elif self.mod == 'ml':

            # if len(seq_types.shape)==3:
            # seq_onehot_types = nn.functional.one_hot(seq_types.sum(-1).bool().long(), num_classes=self.n_cifs+1)[:,:,1:].type(torch.float)    # [B,L,num_classes]
            seq_onehot_types = seq_types
        elif self.mod == 'mc':
            seq_onehot_types = nn.functional.one_hot(
                seq_types, num_classes=self.n_cifs+1)[:, :, 1:].type(torch.float)

        # seq_onehot_types = nn.functional.one_hot(seq_types, num_classes=self.num_types+1)[:,:,1:].type(torch.float)

        dt_seq = (seq_times[:, 1:] - seq_times[:, :-1]) * \
            non_pad_mask[:, 1:]  # [B,L-1]

        # self.start_point = self.start_layer(embed_event) # [B,L,d_in]
        # self.converge_point = self.converge_layer(embed_event) # [B,L,d_in]
        # self.omega = self.decay_layer(embed_event) # [B,L,d_in]

        # # log of intensity
        # cell_t = self.state_decay(self.converge_point[:,1:,:], self.start_point[:,1:,:], self.omega[:,1:,:], dt_seq[:, :, None]) # [B,L-1,d_in]

        self.start_point = self.start_layer(
            embed_event)[:, :-1, :] * non_pad_mask[:, 1:, None]  # [B,L-1,d_in]  1:L-1
        self.converge_point = self.converge_layer(
            embed_event)[:, :-1, :] * non_pad_mask[:, 1:, None]  # [B,L-1,d_in]
        self.omega = self.decay_layer(
            embed_event)[:, :-1, :] * non_pad_mask[:, 1:, None]  # [B,L-1,d_in]
        # log of intensity
        cell_t = self.state_decay(
            self.converge_point, self.start_point, self.omega, dt_seq[:, :, None])  # [B,L-1,d_in]

        # Get the intensity process
        # intens_at_evs = self.intensity_layer(torch.cat([cell_t,cell_t_state],dim=-1)) # [B,L-1,n_cif]
        # + self.intensity_layer_state(cell_t_state) # [B,L-1,n_cif] 2:L
        intens_at_evs = self.intensity_layer(cell_t)
        self.intens_at_evs = intens_at_evs

        # [B,L-1,n_cif] intensity at occurred types
        true_intens_at_evs = intens_at_evs * seq_onehot_types[:, 1:, :]
        intens_at_evs_sumK = torch.sum(true_intens_at_evs, dim=-1)  # [B,L-1]
        intens_at_evs_sumK.masked_fill_(~non_pad_mask[:, 1:].bool(), 1.0)

        log_sum = torch.sum(torch.log(intens_at_evs_sumK), dim=-1)  # [B]
        # integral
        taus = torch.rand(n_batch, n_times, 1, self.n_mc_samples).to(
            device)  # self.process_dim replaced 1  [B,L-1,1,ns]
        taus = dt_seq[:, :, None, None] * taus  # inter-event times samples)

        # sampled_times = taus + seq_times[:, :-1,None, None]

        # cell_tau = self.state_decay(
        #     self.converge_point[:,1:,:,None],
        #     self.start_point[:,1:,:,None],
        #     self.omega[:,1:,:,None],
        #     taus) # [B,L-1,d_model,ns]

        # cell_tau = self.state_decay(
        #     self.converge_point[:,:,:,None] * non_pad_mask[:, 1:,None,None],
        #     self.start_point[:,:,:,None] * non_pad_mask[:, 1:,None,None],
        #     self.omega[:,:,:,None] * non_pad_mask[:, 1:,None,None],
        #     taus) # [B,L-1,d_model,ns]

        cell_tau = self.state_decay(
            self.converge_point[:, :, :, None],
            self.start_point[:, :, :, None],
            self.omega[:, :, :, None],
            taus)  # [B,L-1,d_model,ns]

        cell_tau = cell_tau.transpose(2, 3)  # [B,L-1,ns,d_model]

        # intens_at_samples = self.intensity_layer(torch.cat([cell_tau,cell_tau_state],dim=-1)).transpose(2,3) # [B,L-1,K,ns]
        intens_at_samples = self.intensity_layer(
            cell_tau).transpose(2, 3)  # +\
        #                          self.intensity_layer_state(cell_tau_state).transpose(2,3)# [B,L-1,K,ns]

        intens_at_samples = intens_at_samples * \
            non_pad_mask[:, 1:, None, None]  # [B,L-1,n_cif,ns]
        total_intens_samples = intens_at_samples.sum(
            dim=2)  # shape batch * N * MC  [B,L-1,ns]
        partial_integrals = dt_seq * \
            total_intens_samples.mean(dim=2)  # [B,L-1]

        integral_ = partial_integrals.sum(dim=1)  # [B]

        # ****************************************************** MULTI-LABEL case:

        # intens_at_evs_sumK = torch.sum(true_intens_at_evs, dim=-1) # [B,L-1]
        # intens_at_evs_sumK.masked_fill_(~non_pad_mask[:, 1:].bool(), 1.0)

        if self.mod == 'ml':
            aaa = 1
            p = intens_at_evs * \
                torch.exp(-aaa*partial_integrals[:, :, None]) * \
                non_pad_mask[:, 1:, None]  # [B,L-1,n_cif]
            if p.max() > 0.999:
                p = torch.clamp(p, max=0.99)
                print("WTF")
                a = 1
            one_min_true_log_density = (
                1-seq_onehot_types[:, 1:, :])*torch.log(1-p) * non_pad_mask[:, 1:, None]  # [B,L-1,n_cif]
            log_sum = log_sum + one_min_true_log_density.sum(-1).sum(-1)  # [B]
            if torch.isinf(one_min_true_log_density.sum()):
                print("WTF")
                a = 1

        self.intens_at_samples = intens_at_samples
        self.taus = taus

        self.true_intens_at_evs = true_intens_at_evs

        # res = torch.sum(- log_sum + integral_)
        return log_sum, integral_


class CIF_thp(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, d_in, n_cifs, mod_CIF='MHP_multilabel'):
        super().__init__()

        self.d_in = d_in
        self.d_in = d_in
        self.n_cifs = n_cifs

        self.n_mc_samples = 20
        self.mod == mod_CIF

        # THP decoder
        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.ones(self.d_CIF) * (-0.1))

        # parameter for baseline b_k
        self.base = nn.Parameter(torch.ones(self.d_CIF) * (0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.ones(self.d_CIF) * (1.0))

        # state-dependent base intensity
        if self.state:
            self.base_R = nn.Linear(self.d_DA, self.d_CIF, bias=False)

        # convert hidden vectors into a scalar
        # if self.state:
        self.linear = nn.Linear(self.d_con, self.d_CIF)

    def forward(self, embed_info, seq_times, seq_types, non_pad_mask):

        # event_ll, non_event_ll = opt.event_loss(model, enc_out, event_time, event_type, side = prediction, mod=opt.mod)

        n_batch = seq_times.size(0)
        n_times = seq_times.size(1) - 1  # L-1
        device = seq_times.device

        non_pad_mask = get_non_pad_mask(seq_types).squeeze(2)
        dt_seq = (seq_times[:, 1:] - seq_times[:, :-1]) * \
            non_pad_mask[:, 1:]  # [B,L-1]

        if len(seq_types.shape) == 3:
            # seq_onehot_types = nn.functional.one_hot(seq_types.sum(-1).bool().long(), num_classes=self.num_types+1)[:,:,1:].type(torch.float)    # [B,L,num_classes]
            seq_onehot_types = seq_types
        else:
            seq_onehot_types = nn.functional.one_hot(
                seq_types, num_classes=self.num_types+1)[:, :, 1:].type(torch.float)

        # lambda at timestamps
        all_hid = self.linear(embed_info)  # [B,L,K]
        intens_at_evs = Utils.softplus(
            self.alpha[None, None, :]*dt_seq[:, :, None] + all_hid[:, 1:, :]+self.base[None, None, :], self.beta)  # [B,L-1,K]

        # [B,L-1,K] intensity at occurred types
        true_intens_at_evs = intens_at_evs * seq_onehot_types[:, 1:, :]
        intens_at_evs_sumK = torch.sum(true_intens_at_evs, dim=-1)  # [B,L-1]
        intens_at_evs_sumK.masked_fill_(~non_pad_mask[:, 1:].bool(), 1.0)

        log_sum = torch.sum(torch.log(intens_at_evs_sumK), dim=-1)  # [B]

        # integral

        taus = torch.rand(n_batch, n_times, 1, self.n_mc_samples).to(
            device)  # self.process_dim replaced 1  [B,L-1,1,ns]
        taus = dt_seq[:, :, None, None] * taus  # inter-event times samples)

        intens_at_samples = Utils.softplus(self.alpha[None, None, :, None]*taus+all_hid[:, 1:, :,
                                           None]+self.base[None, None, :, None], self.beta[None, None, :, None])  # [B,L-1,K,ns]
        intens_at_samples = intens_at_samples * \
            non_pad_mask[:, 1:, None, None]  # [B,L-1,K,ns]
        total_intens_samples = intens_at_samples.sum(
            dim=2)  # shape batch * N * MC  [B,L-1,ns]
        partial_integrals = dt_seq * \
            total_intens_samples.mean(dim=2)  # [B,L-1]

        integral_ = partial_integrals.sum(dim=1)  # [B]

        # ****************************************************** MULTI-LABEL case:
        if self.mod == 'MHP_multilabel':

            p = intens_at_evs * \
                torch.exp(-partial_integrals[:, :, None]) * \
                non_pad_mask[:, 1:, None]  # [B,L-1,K]
            if p.max() > 0.999:
                a = 1
            one_min_true_log_density = (
                1-seq_types[:, 1:, :])*torch.log(1-p) * non_pad_mask[:, 1:, None]  # [B,L-1,K]
            log_sum = log_sum + one_min_true_log_density.sum(-1).sum(-1)  # [B]
            if torch.isinf(one_min_true_log_density.sum()):
                a = 1

        # res = torch.sum(- log_sum + integral_)
        return log_sum, integral_


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_mark, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_mark, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_mark)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out



class CumulativeSetAttentionLayer(nn.Module):
    # dense_options = {
    #     'activation': 'relu',
    #     'kernel_initializer': 'he_uniform'
    # }
    def __init__(self,
                 dim_s,
                 n_layers=2, d_psi=128, d_rho=128,
                 aggregation_function='mean',
                 dot_prod_dim=64, n_heads=4, attn_dropout=0.3):
        super().__init__()
        assert aggregation_function == 'mean'
        self.d_psi = d_psi
        self.d_rho = d_rho
        self.dim_s = dim_s

        self.dot_prod_dim = dot_prod_dim
        self.attn_dropout = attn_dropout
        self.n_heads = n_heads
        # self.psi = MLP_state(
        #     n_layers, d_psi, 0., self.dense_options)

        self.psi = MLP_state(dim_s, d_psi, n_layers, 0)
        # self.psi.add(Dense(d_rho, **self.dense_options))

        # self.rho = Dense(d_rho, **self.dense_options)
        self.rho = nn.Linear(d_psi, d_rho)

        # self.cumulative_segment_mean = cumulative_segment_wrapper(cumulative_mean)

        self.W_k = nn.Parameter(torch.rand(
            [self.d_rho+self.dim_s, self.dot_prod_dim*self.n_heads]))
        self.W_q = nn.Parameter(torch.rand([self.n_heads, self.dot_prod_dim]))

        nn.init.xavier_normal_(self.W_k)
        nn.init.xavier_normal_(self.W_q)
        nn.init.xavier_normal_(self.rho.weight)


    def forward(self, inputs, mask, training=None):
        # if training is None:
        #     training = tf.keras.backend.learning_phase()

        encoded = self.psi(inputs, mask)  # [B,P, d_psi]

        # cumulative mean aggregation
        # agg = self.cumulative_segment_mean(encoded, segment_ids)

        # implement cummean
        # across P dimension [B,P, d_psi]
        agg = (torch.cumsum(encoded, dim=1) /
               torch.cumsum(torch.ones_like(encoded), dim=1))*mask
        agg = self.rho(agg) * mask  # [B,P, d_rho] latent_width

        # matrix A [B, P, d_s+d_rho]
        combined = torch.concat([inputs, agg], axis=-1)

        keys = torch.matmul(combined, self.W_k) * mask  # [B, P,dot_prod_dim*m]
        # [B, P,m, dot_prod_dim]
        keys = torch.stack(torch.split(keys, self.n_heads, -1), -1)
        keys = keys.unsqueeze(3)  # torch.expand_dims(keys, axis=2)
        # should have shape (B,P, heads, 1, dot_prod_dim)

        # queries = torch.expand_dims(torch.expand_dims(self.W_q, -1), 0)
        queries = self.W_q.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        # should have shape (1, 1, heads, dot_prod_dim, 1)

        preattn = torch.matmul(
            keys, queries) / torch.sqrt(torch.tensor(self.dot_prod_dim))  # [B, P,m, 1,1]
        preattn = torch.squeeze(torch.squeeze(preattn, -1), -1)  # [B, P,m]
        # [P, heads]

        return preattn * mask  # [B, P,m]


class MLP_state(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, d_in, d_out, n_layers, dropout):
        super(MLP_state, self).__init__()

        # self.module = nn.Sequential([

        self.fc1 = nn.Linear(d_in, d_out)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(0.0)
        self.fc2 = nn.Linear(d_out, d_out)

        # ])
        # self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, data, non_pad_mask):
        # data # [B,P, d_in]
        # non_pad_mask [B,P,1]
        out = self.relu(self.fc1(data * non_pad_mask))  # [B,P, d_out]
        out = self.do(out)
        out = self.relu(self.fc2(out * non_pad_mask))  # [B,P, d_out]
        out = out * non_pad_mask
        return out


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

