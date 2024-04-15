import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import get_non_pad_mask
from transformer import Constants

import numpy as np

def cal_cif(model, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:] # [B,L-1]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device) # [B, L-1,num_samples]
    # temp_time /= (time[:, :-1] + 1).unsqueeze(2)
    # print(time[:, :-1])

    ref_time = time[:, :-1].unsqueeze(-1) + temp_time # [B, L-1,num_samples]
    
    temp_hid = model.linear(data)[:, 1:, :] # [B,L-1,K]
    temp_hid = temp_hid.unsqueeze(2).repeat(1,1,num_samples,1) # [B,L-1,ns,K]
    # temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)  # [B,L,1] 

    all_lambda = softplus(temp_hid + temp_time.unsqueeze(-1) * model.alpha + model.base, model.beta) # [B, L-1,num_samples,K]
    # all_lambda = torch.sum(all_lambda, dim=2) / num_samples # [B,L-1,K] 

    # unbiased_integral = all_lambda * diff_time.unsqueeze(-1) # [B,L-1,K] 
    # unbiased_integral = torch.sum(unbiased_integral, dim=-1) # [B,L-1] 
    return ref_time, all_lambda

def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    # event[event==0.0] = 1.0
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:] # [B,L-1]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device) # [B, L-1,num_samples]
    # temp_time /= (time[:, :-1] + 1).unsqueeze(2)
    # print(time[:, :-1])

    
    
    temp_hid = model.linear(data)[:, 1:, :] # [B,L-1,K]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)  # [B,L,1] 

    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta) # [B, L-1,num_samples]
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples # [B,L-1] # this seems to be wrong!

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral

def HK_compute_integral_unbiased(model, data, time, non_pad_mask, type_mask, side = None):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:] # [B,L-1]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device) # [B, L-1,num_samples]
    # temp_time /= (time[:, :-1] + 1).unsqueeze(2)
    # print(time[:, :-1])

    
    temp_hid = model.linear(data)[:, :-1, :] # [B,L-1,K]
    temp_hid = temp_hid.unsqueeze(2)#.repeat(1,1,num_samples,1) # [B,L-1,1,K]
    # temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)  # [B,L,1] 

    all_lambda = softplus(temp_hid + temp_time.unsqueeze(-1) * model.alpha + model.base, model.beta) # [B, L-1,num_samples,K]
    # all_lambda = softplus(temp_hid + temp_time.unsqueeze(-1) * model.alpha, model.beta)+\
    #             softplus(model.base, model.beta) # [B, L-1,num_samples,K]

    if model.state:
        # side[-1] is r_enc
        # all_lambda = softplus(temp_hid, model.beta) + softplus(model.base_R(side[-1])[:, :-1, :], model.beta) # [B,L,K]  NEW spearete

        all_lambda = softplus(temp_hid + temp_time.unsqueeze(-1) * model.alpha, model.beta)+\
                softplus(model.base_R(side[-1])[:, :-1, None,:], model.beta) # [B, L-1,num_samples,K]




    all_lambda = torch.sum(all_lambda, dim=2) / num_samples # [B,L-1,K] 

    # # # how to save
    # import numpy
    # i=all_lambda[0,:,:,:].reshape(-1,model.num_types) # [L-1 * ns,K]
    # numpy.savetxt('all_lambda.txt', i.cpu().detach().numpy())
    # t = ( (time[0,:-1]-time[0,0])* non_pad_mask[:, :-1] )[0,:,None] + temp_time[0,:,:] # [L-1,ns]
    # numpy.savetxt('temp_time.txt',t.flatten().cpu().detach().numpy())

    unbiased_integral = all_lambda * diff_time.unsqueeze(-1) # [B,L-1,K] 
    # unbiased_integral = torch.sum(unbiased_integral, dim=-1) # [B,L-1] 
    return unbiased_integral
# def log_likelihood(model, data, time, types):
#     """ Log-likelihood of sequence. """

#     non_pad_mask = get_non_pad_mask(types).squeeze(2)

#     type_mask = torch.zeros([*types.size(), model.num_types], device=data.device) # [B,L,K]
#     for i in range(model.num_types):
#         type_mask[:, :, i] = (types == i + 1).bool().to(data.device)
    
#     # type_mask = nn.functional.one_hot(types, num_classes=model.num_types+1)[:,:,1:].type(torch.float)    # [B,L,num_classes]

#     all_hid = model.linear(data)
#     # all_lambda = softplus(all_hid+model.base, model.beta) # [B,L,K]
#     all_lambda = softplus(all_hid, model.beta) # [B,L,K]
#     type_lambda = torch.sum(all_lambda * type_mask, dim=2) # [B,L] only keep occured events

#     # event log-likelihood
#     event_ll = compute_event(type_lambda, non_pad_mask) # [B,L] y_1, y_4, y_2
#     event_ll = torch.sum(event_ll, dim=-1) # [B]

#     # non-event log-likelihood, either numerical integration or MC integration
#     # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
#     non_event_ll = HK_compute_integral_unbiased(model, data, time, non_pad_mask, type_mask)
#     non_event_ll = torch.sum(non_event_ll, dim=-1) # [B]
#     non_event_ll = torch.sum(non_event_ll, dim=-1) # [B]

#     return event_ll, non_event_ll

def log_likelihood(model, data, time, types, w= None, side=None):
    """ Log-likelihood of sequence. """

    non_pad_mask = get_non_pad_mask(types).squeeze(2)

    # type_mask = torch.zeros([*types.size(), model.num_types], device=data.device) # [B,L,K]
    # for i in range(model.num_types):
    #     type_mask[:, :, i] = (types == i + 1).bool().to(data.device)
    
    # type_mask = nn.functional.one_hot(types, num_classes=model.num_types+1)[:,:,1:].type(torch.float)    # [B,L,num_classes]
    if len(types.shape)==3:
        type_mask = nn.functional.one_hot(types.sum(-1).bool().long(), num_classes=model.num_types+1)[:,:,1:].type(torch.float)    # [B,L,num_classes]
    else:
        type_mask = nn.functional.one_hot(types, num_classes=model.num_types+1)[:,:,1:].type(torch.float)
    
    # lambda at timestamps
    all_hid = model.linear(data)
    # all_lambda = softplus(all_hid, model.beta) # [B,L,K] old wrong does not consider baseline
    all_lambda = softplus(all_hid+model.base, model.beta) # [B,L,K]
    # all_lambda = softplus(all_hid, model.beta) + softplus(model.base, model.beta) # [B,L,K]  NEW spearete

    if model.state:
        # side[-1] is r_enc
        all_lambda = softplus(all_hid, model.beta) + softplus(model.base_R(side[-1]), model.beta) # [B,L,K]  NEW spearete




    type_lambda = torch.sum(all_lambda * type_mask, dim=2) # [B,L] only keep occured events
   
    # 
    # all_prob = type_lambda[:,:-1] * torch.exp(-int_lambda) # [B,L-1]
    # all_prob = torch.exp(-int_lambda) # [B,L-1]

    # all_prob = type_lambda[:,:-1] # [B,L-1]
    
    # # NEW notation ************************************************************************
    # labels = type_mask[:,:-1,:]  # one-hot encoding or binary encoding (multi-label case) # [B,L-1,M]
    # log_intensity = torch.log(all_lambda[:,:-1,:])                # [B,L-1,M]
    # intensity_mask = non_pad_mask[:,:-1] # [B,L-1]
    # intensity_integral = HK_compute_integral_unbiased(model, data, time, non_pad_mask, type_mask) # [B,L-1,K] 
    # intensity_integral = torch.sum(intensity_integral, dim=-1)  # [B,L-1]


    # log_density = (log_intensity - intensity_integral.unsqueeze(dim=-1))        # [B,L-1,M]
    # log_density = log_density * intensity_mask.unsqueeze(dim=-1)  # [B,L,M]

    # true_log_density = log_density * labels                       # [B,L,M]
    # true_log_density_flat = true_log_density.reshape(
    # true_log_density.shape[0], -1)                            # [B,L*M]
    # log_likelihood = torch.sum(true_log_density_flat, dim=-1)        # [B]
    
    
    # if model.multi_labels:
    #     eps = epsilon(dtype=log_density.dtype, device=log_density.device)
    #     log_density = torch.clamp(log_density, max=-eps)
    #     one_min_density = 1. - torch.exp(log_density) + eps  # [B,L,M]
    #     log_one_min_density = torch.log(one_min_density)  # [B,L,M]
    #     log_one_min_density = (log_one_min_density *
    #                             intensity_mask.unsqueeze(dim=-1))
    #     one_min_true_log_density = (1. - labels) * log_one_min_density
    #     one_min_true_log_density_flat = one_min_true_log_density.reshape(
    #         one_min_true_log_density.shape[0], -1)  # [B,L*M]
    #     log_likelihood = log_likelihood + torch.sum(
    #         one_min_true_log_density_flat, dim=-1)  # [B]

    
    
    # loss
    # loss = compute_event( all_prob,non_pad_mask[:,:-1]) # [B,L-1]
    # loss = torch.log( all_prob) # [B,L-1]
    # loss = all_prob
    # loss = torch.sum(loss, dim=-1)
    # loss = type_mask * compute_event(all_prob,non_pad_mask) + (1-type_mask) *compute_event(1-all_prob,non_pad_mask)

    
    
    # OLD
     


    # integral between consecutive events
    # int_lambda = HK_compute_integral_unbiased(model, data, time, non_pad_mask, type_mask) # [B,L-1,K]
    # int_lambda = int_lambda.sum(-1) # [B,L-1]
    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask) # [B,L] y_1, y_4, y_2
    event_ll = torch.sum(event_ll, dim=-1) # [B]

    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = HK_compute_integral_unbiased(model, data, time, non_pad_mask, type_mask, side=side) # [B,L-1,K]
    
    if w is not None:
        non_event_ll = non_event_ll * w
    
    non_event_ll = torch.sum(non_event_ll, dim=-1) # [B,L-1]
    non_event_ll = torch.sum(non_event_ll, dim=-1) # [B]


    # modified
    # event_ll = loss
    # non_event_ll = loss*0.0
    # if abs( (torch.sum(event_ll - non_event_ll).item() - torch.sum(loss).item())/torch.sum(loss).item() ) >0.05:
    #     asdas=2
        # print("FUCK")
        # print(-torch.sum(event_ll - non_event_ll).item(), -torch.sum(loss).item())
        # print(-torch.sum(event_ll - non_event_ll).item(), -torch.sum(loss).item())
    return event_ll, non_event_ll
    # return log_likelihood, log_likelihood*0.0
    # print(event_ll - non_event_ll,'\n',log_likelihood)
def type_loss_CE(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    y_true = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    y_pred = torch.max(prediction, dim=-1)[1]
    
    # additional
    masks = y_true>-1
    # y_pred = pred_type # [*]
    # y_true = truth # [*]
    y_score = nn.Softmax(dim=-1)(prediction) # [B,L,C] -> [*,C]
    # y_pred_stupid =  types - 1 # [*]
    
    correct_num = torch.sum((y_pred == y_true)*(masks))

    
    
    
    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, y_true)
    else:
        loss = loss_func(prediction.transpose(1, 2), y_true)

    loss = torch.sum(loss)


    




    return loss, correct_num, (y_pred, y_true, y_score, masks)

def type_loss_BCE(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    y_true = types[:, 1:,:]
    prediction = prediction[:, :-1, :]

    y_pred = prediction.gt(0).type(torch.int)
    # correct_num=5

    # additional
    masks = (y_true.sum(-1).gt(0))[:,:,None] #[B,L-1,1]
    
    
    
    
    
    # y_pred = pred_type # [*]
    # y_true = truth # [*]
    y_score = nn.Sigmoid()(prediction) # [B,L,C] -> [*,C]
    # y_pred_stupid =  types - 1 # [*]
    correct_num = torch.sum((y_pred == y_true)*(masks))

    
    
    
    
    # compute cross entropy loss
    # if isinstance(loss_func, LabelSmoothingLoss):
    #     loss = loss_func(prediction, truth)
    # else:
    loss = loss_func(prediction, y_true.float()) * (masks)

    loss = torch.sum(loss)


    




    return loss, correct_num, (y_pred, y_true, y_score, masks)

def time_loss(prediction, event_time, non_pad_mask=None):
    """ Time prediction loss. """
    # non_pad_mask.squeeze_(-1)
    prediction.squeeze_(-1)
    
    
    # true = event_time[:, 1:] - event_time[:, :-1]
    true = (event_time[:, 1:] - event_time[:, :-1]) * non_pad_mask[:, 1:] # [B,L-1]

    prediction = prediction[:, :-1]

    # event time gap prediction
    diff = prediction - true
    sse = torch.sum(diff * diff)
    sae = torch.sum(torch.abs(diff))

    mask = true == 0
    diff = (prediction[~mask] - true[~mask])/true[~mask]
    sse_norm = torch.sum(diff * diff)
    

    return sse, sse_norm, sae

def time_loss2(prediction, time_gap, h = 1, pt=None):
    """ Time prediction loss.
        time_gap is [B,L] in the form of absolute time
        prediction is [B,L,horizon] in the form of time gap

     """
    # prediction = prediction[:,:,0]
    # true = time_gap[:, h:] - time_gap[:, :-h]   # to compute the time gap
    # new method ***************************************************************************

    hh = 1

    L = prediction.shape[1]
    prediction = prediction.reshape(prediction.shape[0],-1) # [B,L*h]
    true = F.pad(time_gap[:, hh:], (0,hh,0,0),'constant',Constants.PAD) # [B,L,1] add a pad at the end

    for i in range(2,h+1):
        true = torch.cat([true, F.pad(time_gap[:, i:], (0,i,0,0),'constant',Constants.PAD) ], dim=1) 
    # true [B,L,C*h]
    masks = true==Constants.PAD # [B,L,1*h]

    prediction[:,:L][~masks[:,:L]], true[:,:L][~masks[:,:L]]
    diff = prediction[:,:L][~masks[:,:L]] - true[:,:L][~masks[:,:L]]
    se = torch.sum(diff * diff) # sum of squared errors
    ae = ( nn.L1Loss(reduction='none')(prediction[:,:L][~masks[:,:L]],true[:,:L][~masks[:,:L]]) )# sum of absolute errors )
    nae=1


    tgap_true = true[~masks]
    tgap_pred = prediction[~masks]

    if pt is not None:  # only for evaluation
        time_gap_pred_inv_tran = torch.tensor(pt.inverse_transform(prediction[:,:L][~masks[:,:L]].cpu().numpy().reshape(-1,1)).flatten(), requires_grad=True)
        time_gap_true_inv_tran = torch.tensor(pt.inverse_transform(true[:,:L][~masks[:,:L]].cpu().numpy().reshape(-1,1)).flatten(), requires_grad=True)
        diff = time_gap_pred_inv_tran - time_gap_true_inv_tran
        if torch.isnan(diff).sum()>0:
            aaaa=1
        diff[torch.isnan(diff)] = 0
        se = torch.sum(diff * diff) # sum of squared errors
        sae = torch.sum (nn.L1Loss(reduction='none')(time_gap_pred_inv_tran, time_gap_true_inv_tran) ) # sum of absolute errors
        se.requires_grad = True
        sae.requires_grad = True
        tgap_true = time_gap_true_inv_tran
        tgap_pred= time_gap_pred_inv_tran


    
    return se,torch.sum(ae), nae, (tgap_true.detach().cpu(), tgap_pred.detach().cpu())
class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss

# THP

def thp_log_likelihood(model, embed_info, seq_times, seq_types,n_mc_samples=20, w= None, side=None, mod='MHP_multiclass'):
    """ Log-likelihood of sequence. """
    n_batch = seq_times.size(0)
    n_times = seq_times.size(1) - 1 # L-1
    device = seq_times.device

    non_pad_mask = get_non_pad_mask(seq_types).squeeze(2)
    dt_seq = (seq_times[:, 1:] - seq_times[:, :-1]) * non_pad_mask[:, 1:] # [B,L-1]

    if len(seq_types.shape)==3:
        # seq_onehot_types = nn.functional.one_hot(seq_types.sum(-1).bool().long(), num_classes=model.num_types+1)[:,:,1:].type(torch.float)    # [B,L,num_classes]
        seq_onehot_types = seq_types
    else:
        seq_onehot_types = nn.functional.one_hot(seq_types, num_classes=model.num_types+1)[:,:,1:].type(torch.float)
    
    # lambda at timestamps
    all_hid = model.linear(embed_info) # [B,L,K]
    intens_at_evs = softplus(model.alpha[None,None,:]*dt_seq[:,:,None] +all_hid[:,1:,:]+model.base[None,None,:], model.beta) # [B,L-1,K]

    true_intens_at_evs = intens_at_evs * seq_onehot_types[:,1:,:] # [B,L-1,K] intensity at occurred types
    intens_at_evs_sumK = torch.sum(true_intens_at_evs, dim=-1) # [B,L-1]
    intens_at_evs_sumK.masked_fill_(~non_pad_mask[:, 1:].bool(), 1.0)

    log_sum = torch.sum(torch.log(intens_at_evs_sumK ), dim=-1) # [B] 

    # integral

    taus = torch.rand(n_batch, n_times, 1, n_mc_samples).to(device)# model.process_dim replaced 1  [B,L-1,1,ns]
    taus = dt_seq[:, :, None, None] * taus  # inter-event times samples)

    intens_at_samples = softplus(model.alpha[None,None,:,None]*taus+all_hid[:,1:,:,None]+model.base[None,None,:,None], model.beta[None,None,:,None]) # [B,L-1,K,ns]
    intens_at_samples = intens_at_samples * non_pad_mask[:, 1:,None,None] # [B,L-1,K,ns]
    total_intens_samples = intens_at_samples.sum(dim=2)  # shape batch * N * MC  [B,L-1,ns]
    partial_integrals = dt_seq * total_intens_samples.mean(dim=2) # [B,L-1]

    integral_ = partial_integrals.sum(dim=1) # [B]

    # ****************************************************** MULTI-LABEL case:
    if mod=='MHP_multilabel':

        p = intens_at_evs * torch.exp(-partial_integrals[:,:,None]) * non_pad_mask[:, 1:,None] # [B,L-1,K]
        if p.max()>0.999:
            a=1
        one_min_true_log_density=(1-seq_types[:,1:,:])*torch.log(1-p) * non_pad_mask[:, 1:,None] # [B,L-1,K]
        log_sum = log_sum + one_min_true_log_density.sum(-1).sum(-1) # [B]
        if torch.isinf(one_min_true_log_density.sum()):
            a=1
   
    return log_sum, integral_
   


def thp_log_likelihood_test(model, embed_info, seq_times, seq_types,n_mc_samples = 100, side=None, mod='MHP_multiclass'):
    """ Log-likelihood of non-events, using Monte Carlo integration. """
    
    example={'event_time':seq_times,'event_type':seq_types}
    n_batch = seq_times.size(0)
    n_times = seq_times.size(1) - 1 # L-1
    device = seq_times.device

    non_pad_mask = get_non_pad_mask(seq_types).squeeze(2)
    dt_seq = (seq_times[:, 1:] - seq_times[:, :-1]) * non_pad_mask[:, 1:] # [B,L-1]
    example['dt_seq'] = dt_seq


    if len(seq_types.shape)==3:
        # seq_onehot_types = nn.functional.one_hot(seq_types.sum(-1).bool().long(), num_classes=model.num_types+1)[:,:,1:].type(torch.float)    # [B,L,num_classes]
        seq_onehot_types = seq_types
    else:
        seq_onehot_types = nn.functional.one_hot(seq_types, num_classes=model.num_types+1)[:,:,1:].type(torch.float)
    
    example['seq_onehot_types'] = seq_onehot_types

    # lambda at timestamps
    all_hid = model.linear(embed_info) # [B,L,K]
    intens_at_evs = softplus(all_hid[:,1:,:]+model.base, model.beta) # [B,L-1,K]

    intens_at_evs = intens_at_evs * seq_onehot_types[:,1:,:] # [B,L-1,K]
    example['intens_at_evs'] = intens_at_evs

    intens_at_evs = torch.sum(intens_at_evs, dim=-1) # [B,L-1]
    intens_at_evs.masked_fill_(~non_pad_mask[:, 1:].bool(), 1.0)
    log_sum = torch.sum(torch.log(intens_at_evs ), dim=-1) 

    # integral

    taus = torch.rand(n_batch, n_times, 1, n_mc_samples).to(device)# model.process_dim replaced 1  [B,L-1,1,ns]
    taus = dt_seq[:, :, None, None] * taus  # inter-event times samples)
    example['taus'] = taus

    intens_at_samples = softplus(model.alpha[None,None,:,None]*taus+all_hid[:,1:,:,None]+model.base[None,None,:,None], model.beta[None,None,:,None]) # [B,L-1,K,ns]
    intens_at_samples = intens_at_samples * non_pad_mask[:, 1:,None,None] # [B,L-1,K,ns]
    example['intens_at_samples'] = intens_at_samples

    total_intens_samples = intens_at_samples.sum(dim=2)  # shape batch * N * MC  [B,L-1,ns]
    partial_integrals = dt_seq * total_intens_samples.mean(dim=2) # [B,L-1]

    integral_ = partial_integrals.sum(dim=1) # [B]

    return example



# SAHP

def state_decay(converge_point, start_point, omega, duration_t):
    # * element-wise product
    cell_t = torch.tanh(converge_point + (start_point - converge_point) * torch.exp(- omega * duration_t))
    # cell_t = (converge_point + (start_point - converge_point) * torch.exp(- omega * duration_t))

    return cell_t
def state_decay_states(converge_point, start_point, omega, duration_t, seq_times, state_times):
    # * element-wise product
    mask_event = seq_times>0 # [B,L]
    mask_state = state_times>0 # [B,P]
    mask = mask_event[:,1:,None] * mask_state[:,None,:] # [B,L-1,P]
    if len(duration_t.shape)==2:
        # converge_point, start_point, omega [B,P,d_model]
        # duration_t [B,L-1]
        # seq_times [B,L]
        # state_times [B,P]
        sampled_times = duration_t + seq_times[:, :-1] # [B,L-1]
        duration_state = sampled_times[:,:,None] - state_times[:,None,:] # [B,L-1,P]
        # duration_state = duration_state
        mask_neg_dt = (duration_state>=0) # [B,L-1,P]
        mask_total = (mask * mask_neg_dt)

        exp_comp = - omega[:,None,:,:] * duration_state[:,:,:,None] # [B,L-1,P,d_model]
        exp_comp = exp_comp.masked_fill(~mask_total[:,:,:,None], -1e9)
        cell_t = torch.tanh( start_point[:,None,:,:]  * torch.exp(  exp_comp   )   ) # [B,L-1,P,d_model]
        cell_t = cell_t.sum(dim=-2) # [B,L-1,d_model]
    else:
        # converge_point, start_point, omega [B,P,d_model]
        # duration_t [B,L-1,1,ns]
        # seq_times [B,L]
        # state_times [B,P]
        sampled_times = duration_t + seq_times[:, :-1,None, None] # [B,L-1,1,ns]
        duration_state = sampled_times[:,:,:,:,None] - state_times[:,None,None,None,:] # [B,L-1,1,ns,P]
        # duration_state[duration_state<0]=-200
        mask_neg_dt = (duration_state>=0) # [B,L-1,1,ns,P]
        mask_total = (mask[:,:,None,None,:] * mask_neg_dt) # [B,L-1,1,ns,P]

        exp_comp = -omega[:,None,None,None,:,:]  * duration_state[:,:,:,:,:,None] # [B,L-1,P,d_model] # [B,L-1,1,ns,P,d_model]
        exp_comp = exp_comp.masked_fill(~mask_total[:,:,:,:,:,None], -1e9)

        cell_t = torch.tanh( start_point[:,None,None,None,:,:]  * torch.exp(exp_comp) ) # [B,L-1,1,ns,P,d_model]
        cell_t = cell_t.squeeze(2).sum(dim=-2).transpose(-1,-2) # [B,L-1,d_model,ns]

    # cell_t = torch.tanh(converge_point + (start_point - converge_point) * torch.exp(- omega * duration_t))
    # cell_t = (converge_point + (start_point - converge_point) * torch.exp(- omega * duration_t))

    return cell_t 

def sahp_log_likelihood(model, embed_info, seq_times, seq_types,n_mc_samples = 20, mod='MHP_multiclass'):
    # embed_info [B,L,d_model]
    # seq_times, seq_types [B,L]

    non_pad_mask = get_non_pad_mask(seq_types).squeeze(2)
    
    if len(seq_types.shape)==3:
        # seq_onehot_types = nn.functional.one_hot(seq_types.sum(-1).bool().long(), num_classes=model.num_types+1)[:,:,1:].type(torch.float)    # [B,L,num_classes]
        seq_onehot_types = seq_types
    else:
        seq_onehot_types = nn.functional.one_hot(seq_types, num_classes=model.num_types+1)[:,:,1:].type(torch.float)
        
    # seq_onehot_types = nn.functional.one_hot(seq_types, num_classes=model.num_types+1)[:,:,1:].type(torch.float)
    
    model.start_point = model.start_layer(embed_info) # [B,L,K]
    model.converge_point = model.converge_layer(embed_info) # [B,L,K]
    model.omega = model.decay_layer(embed_info) # [B,L,K]
    
    
    # log of intensity
    dt_seq = (seq_times[:, 1:] - seq_times[:, :-1]) * non_pad_mask[:, 1:] # [B,L-1]
    cell_t = state_decay(model.converge_point[:,1:,:], model.start_point[:,1:,:], model.omega[:,1:,:], dt_seq[:, :, None]) # event 2:L
    # [B,L-1,K]
    
    n_batch = seq_times.size(0)
    n_times = seq_times.size(1) - 1 # L-1
    device = dt_seq.device
    # Get the intensity process
    intens_at_evs = model.intensity_layer(cell_t) # [B,L-1,K] from 2 to L


    
    true_intens_at_evs = intens_at_evs * seq_onehot_types[:,1:,:] # [B,L-1,K] intensity at occurred types
    intens_at_evs_sumK = torch.sum(true_intens_at_evs, dim=-1) # [B,L-1]
    intens_at_evs_sumK.masked_fill_(~non_pad_mask[:, 1:].bool(), 1.0)

    log_sum = torch.sum(torch.log(intens_at_evs_sumK ), dim=-1) # [B]

    # integral
    taus = torch.rand(n_batch, n_times, 1, n_mc_samples).to(device)# model.process_dim replaced 1  [B,L-1,1,ns]
    taus = dt_seq[:, :, None, None] * taus  # inter-event times samples)

    cell_tau = state_decay(
        model.converge_point[:,1:,:,None],
        model.start_point[:,1:,:,None],
        model.omega[:,1:,:,None],
        taus) # [B,L-1,d_model,ns]
    

    cell_tau = cell_tau.transpose(2, 3)  # [B,L-1,ns,d_model]
    intens_at_samples = model.intensity_layer(cell_tau).transpose(2,3) # [B,L-1,K,ns]
    # intens_at_samples = nn.utils.rnn.pad_sequence(
    #     intens_at_samples, padding_value=0.0, batch_first=True)
    intens_at_samples = intens_at_samples * non_pad_mask[:, 1:,None,None] # [B,L-1,K,ns]
    total_intens_samples = intens_at_samples.sum(dim=2)  # shape batch * N * MC  [B,L-1,ns]
    partial_integrals = dt_seq * total_intens_samples.mean(dim=2) # [B,L-1]
    integral_ = partial_integrals.sum(dim=1) # [B]

    # ****************************************************** MULTI-LABEL case:
    if mod=='MHP_multilabel':

        p = intens_at_evs * torch.exp(-partial_integrals[:,:,None]) * non_pad_mask[:, 1:,None] # [B,L-1,K]
        if p.max()>0.999:
            a=1
        one_min_true_log_density=(1-seq_types[:,1:,:])*torch.log(1-p) * non_pad_mask[:, 1:,None] # [B,L-1,K]
        log_sum = log_sum + one_min_true_log_density.sum(-1).sum(-1) # [B]
        if torch.isinf(one_min_true_log_density.sum()):
            a=1


    

    
    
        pass
    # res = torch.sum(- log_sum + integral_)
    return log_sum, integral_


def CIF_sahp(model, embed_info, seq_times, seq_types,n_mc_samples = 20, side=None, mod='MHP_multiclass'):
    # embed_info [B,L,d_model]
    # seq_times, seq_types [B,L]
    n_batch = seq_times.size(0)
    n_times = seq_times.size(1) - 1 # L-1
    device = seq_times.device

    # embed_event = side[-4] # [B,L,d_model]
    # embed_state = side[-2] # [B,L,d_r]
    # embed_event = embed_info[:,:,:model.d_TE] # [B,L,d_model]
    embed_event = embed_info # [B,L,d_model]

    embed_state = embed_info[:,:,model.d_TE:] # [B,L,d_r]
    # state_times = side[-1] # [B,P]
    # embed_state = side[-3] # [B,P,d_r]

    non_pad_mask = get_non_pad_mask(seq_types).squeeze(2)
    if len(seq_types.shape)==3:
        seq_onehot_types = nn.functional.one_hot(seq_types.sum(-1).bool().long(), num_classes=model.num_types+1)[:,:,1:].type(torch.float)    # [B,L,num_classes]
    else:
        seq_onehot_types = nn.functional.one_hot(seq_types, num_classes=model.num_types+1)[:,:,1:].type(torch.float)
      
    # seq_onehot_types = nn.functional.one_hot(seq_types, num_classes=model.num_types+1)[:,:,1:].type(torch.float)
    
    model.start_point = model.start_layer(embed_event) # [B,L,K]
    model.converge_point = model.converge_layer(embed_event) # [B,L,K]
    model.omega = model.decay_layer(embed_event) # [B,L,K]
    
    # model.start_point_state = model.start_layer_state(embed_state_red) # [B,L,K]
    # model.converge_point_state = model.converge_layer_state(embed_state_red) # [B,L,K]
    # model.omega_state = model.decay_layer_state(embed_state_red) # [B,L,K]    

    # log of intensity
    dt_seq = (seq_times[:, 1:] - seq_times[:, :-1]) * non_pad_mask[:, 1:] # [B,L-1]
    cell_t = state_decay(model.converge_point[:,1:,:], model.start_point[:,1:,:], model.omega[:,1:,:], dt_seq[:, :, None]) 

    
    # cell_t_state = state_decay(
    #     model.converge_layer_state(embed_state)[:,1:,:]*0,
    #     model.start_layer_state(embed_state)[:,1:,:],
    #     model.decay_layer_state(embed_state)[:,1:,:],
    #     dt_seq[:, :, None],
    #     # seq_times,
    #     # state_times
    # )    # event 2:L
    # # [B,L-1,K]
    

    
    # Get the intensity process
    # intens_at_evs = model.intensity_layer(torch.cat([cell_t,cell_t_state],dim=-1)) # [B,L-1,K]
    intens_at_evs = model.intensity_layer(cell_t) #+ model.intensity_layer_state(cell_t_state) # [B,L-1,K]

    true_intens_at_evs = intens_at_evs * seq_onehot_types[:,1:,:] # [B,L-1,K] intensity at occurred types
    intens_at_evs_sumK = torch.sum(true_intens_at_evs, dim=-1) # [B,L-1]
    intens_at_evs_sumK.masked_fill_(~non_pad_mask[:, 1:].bool(), 1.0)

    log_sum = torch.sum(torch.log(intens_at_evs_sumK ), dim=-1) # [B]

    # integral
    taus = torch.rand(n_batch, n_times, 1, n_mc_samples).to(device)# model.process_dim replaced 1  [B,L-1,1,ns]
    taus = dt_seq[:, :, None, None] * taus  # inter-event times samples)
    
    # sampled_times = taus + seq_times[:, :-1,None, None]

    cell_tau = state_decay(
        model.converge_point[:,1:,:,None],
        model.start_point[:,1:,:,None],
        model.omega[:,1:,:,None],
        taus) # [B,L-1,d_model,ns]
    cell_tau = cell_tau.transpose(2, 3)  # [B,L-1,ns,d_model]

    # cell_tau_state = state_decay(
    #     model.converge_layer_state(embed_state)[:,1:,:,None]*0,
    #     model.start_layer_state(embed_state)[:,1:,:,None],
    #     model.decay_layer_state(embed_state)[:,1:,:,None],
    #     taus,
    #     # seq_times,
    #     # state_times
    # ) # [B,L-1,d_model,ns]
    # cell_tau_state = cell_tau_state.transpose(2, 3)  # [B,L-1,ns,d_model]


    # intens_at_samples = model.intensity_layer(torch.cat([cell_tau,cell_tau_state],dim=-1)).transpose(2,3) # [B,L-1,K,ns]
    intens_at_samples = model.intensity_layer(cell_tau).transpose(2,3)#+\
    #                          model.intensity_layer_state(cell_tau_state).transpose(2,3)# [B,L-1,K,ns]


    intens_at_samples = intens_at_samples * non_pad_mask[:, 1:,None,None] # [B,L-1,K,ns]
    total_intens_samples = intens_at_samples.sum(dim=2)  # shape batch * N * MC  [B,L-1,ns]
    partial_integrals = dt_seq * total_intens_samples.mean(dim=2) # [B,L-1]

    integral_ = partial_integrals.sum(dim=1) # [B]


    # ****************************************************** MULTI-LABEL case:
    if mod=='MHP_multilabel':

        p = intens_at_evs * torch.exp(-partial_integrals[:,:,None]) * non_pad_mask[:, 1:,None] # [B,L-1,K]
        if p.max()>0.999:
            a=1
        one_min_true_log_density=(1-seq_types[:,1:,:])*torch.log(1-p) * non_pad_mask[:, 1:,None] # [B,L-1,K]
        log_sum = log_sum + one_min_true_log_density.sum(-1).sum(-1) # [B]
        if torch.isinf(one_min_true_log_density.sum()):
            a=1


    


    # res = torch.sum(- log_sum + integral_)
    return log_sum, integral_


def sahp_state_log_likelihood_test(model, embed_info, seq_times, seq_types,n_mc_samples = 100, side=None, mod='MHP_multiclass'):
    # embed_info [B,L,d_model]
    # seq_times, seq_types [B,L]
    n_batch = seq_times.size(0)
    n_times = seq_times.size(1) - 1 # L-1
    device = seq_times.device
    example={'event_time':seq_times,'event_type':seq_types}

    # embed_event = side[-4] # [B,L,d_model]
    # embed_state = side[-2] # [B,L,d_r]
    # embed_event = embed_info[:,:,:model.d_TE] # [B,L,d_model]
    embed_event = embed_info
    embed_state = embed_info[:,:,model.d_TE:] # [B,L,d_r]
    # state_times = side[-1] # [B,P]
    # embed_state = side[-3] # [B,P,d_r]

    non_pad_mask = get_non_pad_mask(seq_types).squeeze(2)
    if len(seq_types.shape)==3:
        seq_onehot_types = nn.functional.one_hot(seq_types.sum(-1).bool().long(), num_classes=model.num_types+1)[:,:,1:].type(torch.float)    # [B,L,num_classes]
    else:
        seq_onehot_types = nn.functional.one_hot(seq_types, num_classes=model.num_types+1)[:,:,1:].type(torch.float)
    
    # seq_onehot_types = nn.functional.one_hot(seq_types, num_classes=model.num_types+1)[:,:,1:].type(torch.float)
    example['seq_onehot_types'] = seq_onehot_types

    model.start_point = model.start_layer(embed_event) # [B,L,K]
    model.converge_point = model.converge_layer(embed_event) # [B,L,K]
    model.omega = model.decay_layer(embed_event) # [B,L,K]
    example['start_point'] = model.start_point
    example['converge_point'] = model.converge_point
    example['omega'] = model.omega

    # model.start_point_state = model.start_layer_state(embed_state_red) # [B,L,K]
    # model.converge_point_state = model.converge_layer_state(embed_state_red) # [B,L,K]
    # model.omega_state = model.decay_layer_state(embed_state_red) # [B,L,K]    

    # log of intensity
    dt_seq = (seq_times[:, 1:] - seq_times[:, :-1]) * non_pad_mask[:, 1:] # [B,L-1]
    example['dt_seq'] = dt_seq

    cell_t = state_decay(model.converge_point[:,1:,:], model.start_point[:,1:,:], model.omega[:,1:,:], dt_seq[:, :, None]) 

    
    # cell_t_state = state_decay(
    #     model.converge_layer_state(embed_state)[:,1:,:]*0,
    #     model.start_layer_state(embed_state)[:,1:,:],
    #     model.decay_layer_state(embed_state)[:,1:,:],
    #     dt_seq[:, :, None],
    #     # seq_times,
    #     # state_times
    # )    # event 2:L
    # # [B,L-1,K]
    

    
    # Get the intensity process
    # intens_at_evs = model.intensity_layer(torch.cat([cell_t,cell_t_state],dim=-1)) # [B,L-1,K]
    intens_at_evs = model.intensity_layer(cell_t)#+ model.intensity_layer_state(cell_t_state) # [B,L-1,K]

    intens_at_evs = intens_at_evs * seq_onehot_types[:,1:,:] # [B,L-1,K]
    example['intens_at_evs'] = intens_at_evs
    # example['intens_at_evs_only'] = model.intensity_layer(cell_t) * seq_onehot_types[:,1:,:]
    # example['intens_at_evs_state'] =  model.intensity_layer_state(cell_t_state) * seq_onehot_types[:,1:,:]

    intens_at_evs = torch.sum(intens_at_evs, dim=-1) # [B,L-1]
    intens_at_evs.masked_fill_(~non_pad_mask[:, 1:].bool(), 1.0)

    log_sum = torch.sum(torch.log(intens_at_evs ), dim=-1) 

    # integral
    taus = torch.rand(n_batch, n_times, 1, n_mc_samples).to(device)# model.process_dim replaced 1  [B,L-1,1,ns]
    taus = dt_seq[:, :, None, None] * taus  # inter-event times samples)
    example['taus'] = taus

    # sampled_times = taus + seq_times[:, :-1,None, None]

    cell_tau = state_decay(
        model.converge_point[:,1:,:,None],
        model.start_point[:,1:,:,None],
        model.omega[:,1:,:,None],
        taus) # [B,L-1,d_model,ns]
    cell_tau = cell_tau.transpose(2, 3)  # [B,L-1,ns,d_model]

    # cell_tau_state = state_decay(
    #     model.converge_layer_state(embed_state)[:,1:,:,None]*0,
    #     model.start_layer_state(embed_state)[:,1:,:,None],
    #     model.decay_layer_state(embed_state)[:,1:,:,None],
    #     taus,
    #     # seq_times,
    #     # state_times
    # ) # [B,L-1,d_model,ns]
    # cell_tau_state = cell_tau_state.transpose(2, 3)  # [B,L-1,ns,d_model]


    # intens_at_samples = model.intensity_layer(torch.cat([cell_tau,cell_tau_state],dim=-1)).transpose(2,3) # [B,L-1,K,ns]
    intens_at_samples = model.intensity_layer(cell_tau).transpose(2,3)#+\
    #                         model.intensity_layer_state(cell_tau_state).transpose(2,3)# [B,L-1,K,ns]

    intens_at_samples = intens_at_samples * non_pad_mask[:, 1:,None,None] # [B,L-1,K,ns]
    example['intens_at_samples'] = intens_at_samples
    # example['intens_at_samples_only'] = model.intensity_layer(cell_tau).transpose(2,3) * non_pad_mask[:, 1:,None,None]
    # example['intens_at_samples_state'] =  model.intensity_layer_state(cell_tau_state).transpose(2,3) * non_pad_mask[:, 1:,None,None]

    total_intens_samples = intens_at_samples.sum(dim=2)  # shape batch * N * MC  [B,L-1,ns]
    partial_integrals = dt_seq * total_intens_samples.mean(dim=2) # [B,L-1]

    integral_ = partial_integrals.sum(dim=1) # [B]

    # res = torch.sum(- log_sum + integral_)
    return example



def sahp_log_likelihood_test(model, embed_info, seq_times, seq_types,n_mc_samples = 100, side=None, mod='MHP_multiclass'):
    # embed_info [B,L,d_model]
    # seq_times, seq_types [B,L]
    # # embed_info = side[-2]

    example={'event_time':seq_times,'event_type':seq_types}
    non_pad_mask = get_non_pad_mask(seq_types).squeeze(2)
    if len(seq_types.shape)==3:
        seq_onehot_types = nn.functional.one_hot(seq_types.sum(-1).bool().long(), num_classes=model.num_types+1)[:,:,1:].type(torch.float)    # [B,L,num_classes]
    else:
        seq_onehot_types = nn.functional.one_hot(seq_types, num_classes=model.num_types+1)[:,:,1:].type(torch.float)
    
    # seq_onehot_types = nn.functional.one_hot(seq_types, num_classes=model.num_types+1)[:,:,1:].type(torch.float)
    example['seq_onehot_types'] = seq_onehot_types

    model.start_point = model.start_layer(embed_info) # [B,L,K]
    model.converge_point = model.converge_layer(embed_info) # [B,L,K]
    model.omega = model.decay_layer(embed_info) # [B,L,K]
    if model.state:
        model.start_point_state = model.start_layer_state(side[-1]) # [B,L,K]
        model.converge_point_state = model.converge_layer_state(side[-1]) # [B,L,K]
        model.omega_state = model.decay_layer_state(side[-1]) # [B,L,K] 
    example['start_point'] = model.start_point
    example['converge_point'] = model.converge_point
    example['omega'] = model.omega

    
    # model.start_point = model.start_layer(embed_info)
    # model.converge_point = model.converge_layer(embed_info)
    # model.omega = model.decay_layer(embed_info)

    # log of intensity
    dt_seq = (seq_times[:, 1:] - seq_times[:, :-1]) * non_pad_mask[:, 1:] # [B,L-1]
    example['dt_seq'] = dt_seq
    cell_t = state_decay(model.converge_point[:,1:,:], model.start_point[:,1:,:], model.omega[:,1:,:], dt_seq[:, :, None]) # event 2:L # [B,L-1,d_model]
    if model.state:
        cell_t_state = state_decay(model.converge_point_state[:,1:,:], model.start_point_state[:,1:,:], model.omega_state[:,1:,:], dt_seq[:, :, None]) # event 2:L
        intens_at_evs_state = model.intensity_layer(cell_t_state) # [B,L-1,K]
        intens_at_evs_state = intens_at_evs_state * seq_onehot_types[:,1:,:] # [B,L-1,K]
        example['intens_at_evs_state'] = intens_at_evs_state        
        
        intens_at_evs_only = model.intensity_layer(cell_t) # [B,L-1,K]
        intens_at_evs_only = intens_at_evs_only * seq_onehot_types[:,1:,:] # [B,L-1,K]
        example['intens_at_evs_only'] = intens_at_evs_only        
        
        cell_t = cell_t + cell_t_state

    n_batch = seq_times.size(0)
    n_times = seq_times.size(1) - 1 # L-1
    device = dt_seq.device
    # Get the intensity process
    intens_at_evs = model.intensity_layer(cell_t) # [B,L-1,K]
    intens_at_evs = intens_at_evs * seq_onehot_types[:,1:,:] # [B,L-1,K]
    example['intens_at_evs'] = intens_at_evs

    intens_at_evs = torch.sum(intens_at_evs, dim=-1) # [B,L-1]
    intens_at_evs.masked_fill_(~non_pad_mask[:, 1:].bool(), 1.0)

    # intens_at_evs = nn.utils.rnn.pad_sequence(
    #     intens_at_evs, padding_value=1.0,batch_first=True)  # pad with 0 to get rid of the non-events, log1=0
    # log_intensities = intens_at_evs.log()  # log intensities
    # seq_mask = seq_onehot_types[:, 1:]
    # log_sum = (log_intensities * seq_mask).sum(dim=(2, 1))  # shape batch
    log_sum = torch.sum(torch.log(intens_at_evs ), dim=-1) 

    # integral
    taus = torch.rand(n_batch, n_times, 1, n_mc_samples).to(device)# model.process_dim replaced 1  [B,L-1,1,ns]
    taus = dt_seq[:, :, None, None] * taus  # inter-event times samples)
    example['taus'] = taus

    cell_tau = state_decay(
        model.converge_point[:,1:,:,None],
        model.start_point[:,1:,:,None],
        model.omega[:,1:,:,None],
        taus) # [B,L-1,d_model,ns]
    if model.state:
        cell_tau_state = state_decay(
                        model.converge_point_state[:,1:,:,None],
                        model.start_point_state[:,1:,:,None],
                        model.omega_state[:,1:,:,None],
                    taus) # [B,L-1,d_model,ns]

        cell_tau_state = cell_tau_state.transpose(2, 3)  # [B,L-1,ns,d_model]
        intens_at_samples_state = model.intensity_layer(cell_tau_state).transpose(2,3) # [B,L-1,K,ns]
        intens_at_samples_state = intens_at_samples_state * non_pad_mask[:, 1:,None,None] # [B,L-1,K,ns]
        example['intens_at_samples_state'] = intens_at_samples_state

        cell_tau_only = cell_tau.transpose(2, 3)  # [B,L-1,ns,d_model]
        intens_at_samples_only = model.intensity_layer(cell_tau_only).transpose(2,3) # [B,L-1,K,ns]
        intens_at_samples_only = intens_at_samples_only * non_pad_mask[:, 1:,None,None] # [B,L-1,K,ns]
        example['intens_at_samples_only'] = intens_at_samples_only




        cell_tau = cell_tau+\
            state_decay(
                model.converge_point_state[:,1:,:,None],
                model.start_point_state[:,1:,:,None],
                model.omega_state[:,1:,:,None],
            taus) # [B,L-1,d_model,ns]

            
    cell_tau = cell_tau.transpose(2, 3)  # [B,L-1,ns,d_model]
    intens_at_samples = model.intensity_layer(cell_tau).transpose(2,3) # [B,L-1,K,ns]
    # intens_at_samples = nn.utils.rnn.pad_sequence(
    #     intens_at_samples, padding_value=0.0, batch_first=True)

    intens_at_samples = intens_at_samples * non_pad_mask[:, 1:,None,None] # [B,L-1,K,ns]
    example['intens_at_samples'] = intens_at_samples

    total_intens_samples = intens_at_samples.sum(dim=2)  # shape batch * N * MC  [B,L-1,ns]
    partial_integrals = dt_seq * total_intens_samples.mean(dim=2) # [B,L-1]

    integral_ = partial_integrals.sum(dim=1) # [B]

    # res = torch.sum(- log_sum + integral_)






    return example



def state_label_loss(state_label,prediction, non_pad_mask, loss_fun):
    # prediction [B,1], state_label [B,L,1]


    lens = non_pad_mask.squeeze(-1).sum(-1).long()
    
    # temp = torch.unbind(prediction)
    # prediction_last = torch.stack([temp[i][lens[i].item()-1] for i in range (len(temp))],0)
    prediction_last = prediction[np.arange(prediction.shape[0])   ,  lens-1,   :] #[B,1]
    y_true = state_label[np.arange(state_label.shape[0])   ,  lens-1,   :] # [B, 1]
    y_pred = prediction_last.gt(0).type(torch.int).squeeze(-1) # [B]
    y_score = nn.Sigmoid()(prediction_last.squeeze(-1)) # [B]

    w = y_true.sum()*0+1
    w_pos = y_true.sum()*0+0.2
    # loss = nn.BCEWithLogitsLoss(weight=w, reduction='none',pos_weight=w_pos)(prediction_last, y_true.float()) 
    loss = loss_fun(prediction_last, y_true.float()) 



    # state_label.sum(1).bool()

    # # # y_true = state_label.sum(1).bool().squeeze(-1).float() # [B]
    # # # # y_true = state_label[:,-1,:].bool().squeeze(-1).float() # [B]

    # # # y_pred = prediction.gt(0).type(torch.int).squeeze(-1) # [B]
    # # # y_score = nn.Sigmoid()(prediction.squeeze(-1))

    # # # w = y_true.sum()*0+1
    # # # loss = nn.BCELoss(weight=w, reduction='none')(y_score, y_true)







    # y_true = state_label.bool().squeeze(-1).float()[:,:-1] # [B,L-1]

    # y_pred = prediction.gt(0).type(torch.int).squeeze(-1)[:,:-1] # [B,L]
    # y_score = nn.Sigmoid()(prediction.squeeze(-1))[:,:-1]

    # w = y_true.sum()*0+1
    # loss = nn.BCEWithLogitsLoss(weight=w, reduction='none')(prediction.squeeze()[:,1:], y_true) * non_pad_mask[:,1:]



    # y_true = state_label[:,1:,:]
    # y_pred = prediction[:,:-1,:].gt(0).type(torch.int)
    # y_score = nn.Sigmoid()(prediction[:,:-1,:])
    # embed_state = prediction[-2] # [B,L,d_r]
    # w = y_true.sum()*0+100
    # loss = nn.BCEWithLogitsLoss(weight=w, reduction='none')(prediction[:,:-1,:], state_label[:,1:,:].float()) * non_pad_mask[:,1:].unsqueeze(-1)



    return loss.sum(), (y_pred, y_true, y_score)