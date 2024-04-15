import sys
from torch.profiler import profile, record_function, ProfilerActivity
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
import optuna
import argparse
import numpy as np
import pandas as pd
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader3

from transformer.Models import Transformer, ATHP, align
from tqdm import tqdm

import datetime
import os
import glob
import re
import shutil

import plotly.express as px

import wandb
from dotenv import load_dotenv


# Project is specified by <entity/project-name>
def dl_runs(all_runs, selected_tag=None):

    summary_list, config_list, name_list, path_list = [], [], [], []
    for run in all_runs:

        if (selected_tag not in run.tags) and (selected_tag is not None):
            continue
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.

        config_list.append(
            {k: v for k, v in run.config.items()
             if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        path_list.append(run.path)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list,
        "path": path_list

    })

    # runs_df.to_csv("project.csv")
    return runs_df


# import custom libraries
# sys.path.append("C:\\DATA\\Tasks\\lib\\hk")
# # import hk_psql
# import hk_utils


def write_to_summary(dict_metrics, opt, i_epoch=-1, prefix=''):
    TSNE_LIMIT = 1000
    all_colors = ["#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
                  "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
                  "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
                  "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
                  "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
                  "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
                  "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9",
                  "#52ef99", "#1c875c", "#69c8c1", "#4c707b", "#b6c5f5", "#1642cd", "#fb5de7", "#be64a7", "#62385e", "#edb1ff", "#860967", "#3d84e3", "#c5df72", "#6e3901", "#fba55c", "#9f2114", "#36e515", "#65a10e", "#474a09", "#f5603a", "#fa217f", "#4007d9"]

    if 'ConfMat' in dict_metrics:
        # fig = plt.figure(figsize=(8,8))
        fig, ax = plt.subplots(figsize=(10, 10))
        dict_metrics['ConfMat'].plot(ax=ax)
        # dict_metrics['ConfMat'].plot()
        # opt.writer.add_figure('matplotlib', fig, i_epoch)
        dict_metrics.pop('ConfMat')
        # plt.close()

    if 'time_gap_data' in dict_metrics:
        fig, ax = plt.subplots(figsize=(10, 10))
        _ = ax.scatter(dict_metrics['time_gap_data'][0]
                       [1:], dict_metrics['time_gap_data'][1][:-1])
        _ = ax.plot([0, 2], [0, 2], 'r-')
        # opt.writer.add_figure('time_gap', fig, i_epoch)
        dict_metrics.pop('time_gap_data')
        # plt.close()

    if 'tsne' in dict_metrics:

        if (opt.i_epoch) % opt.write2tsne == 0:
            with open(f"{opt.add_model}tsne_epoch_{i_epoch}.pkl", 'wb') as handle:
                pickle.dump(dict_metrics['tsne'], handle)

            tsne = TSNE(n_components=2, perplexity=15,
                        learning_rate=10, n_jobs=4)

            if (opt.i_epoch) % opt.write2tsne == 0:
                # with hk_utils.Timer('TSNE COMPUTATIOIN:'):
                # X_tsne = tsne.fit_transform(X_enc)
                print('TSNE')
                X_tsne = tsne.fit_transform(
                    dict_metrics['tsne']['X_enc'][:TSNE_LIMIT, :])

            colors_tsne = [all_colors[label]
                           for label in dict_metrics['tsne']['y_true'][:TSNE_LIMIT]]
            fig, ax = plt.subplots(figsize=(10, 10))
            _ = ax.scatter(X_tsne[:TSNE_LIMIT, 0],
                           X_tsne[:TSNE_LIMIT, 1], c=colors_tsne)

            # _ = ax.scatter(dict_metrics['tsne']['X_tsne'][:,0], dict_metrics['tsne']['X_tsne'][:,1], c=colors_tsne)
            # opt.writer.add_figure('tsne', fig, i_epoch)
        dict_metrics.pop('tsne')
    if 'pred_label/PR_curve' in dict_metrics:
        # fig = plt.figure(figsize=(8,8))
        # fig, ax = plt.subplots(figsize=(10, 10))

        wandb.log(
            {'Precision-Recall Curve': wandb.Image(dict_metrics['pred_label/PR_curve'])}, step=i_epoch)
        plt.close()
        # dict_metrics['pred_label/PR_curve'].plot(ax=ax)
        # dict_metrics['pred_label/PR_curve'].plot()
        # opt.writer.add_figure('matplotlib', fig, i_epoch)
        dict_metrics.pop('pred_label/PR_curve')
        # plt.close()
    for k, v in dict_metrics.items():

        if isinstance(v, np.ndarray):
            # opt.writer.add_histogram(prefix+k, v, i_epoch)
            a = 1
        else:
            # opt.writer.add_scalar(prefix+k, v, i_epoch)
            a = 1

    if opt.wandb:

        wandb.log(
            {(prefix+k): v for k, v in dict_metrics.items()}, step=i_epoch)


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data_event(name, dict_name):
        """ Load data and prepare dataloader for event data """
        additional_info = {}

        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')

            if 'dim_process' in data:
                additional_info['num_types'] = data['dim_process']
            if 'num_marks' in data:
                additional_info['num_marks'] = data['num_marks']
            if 'dict_map_events' in data:
                additional_info['dict_map_events'] = data['dict_map_events']
            if 'pos_weight' in data:
                additional_info['pos_weight'] = data['pos_weight']
            if 'w_class' in data:
                additional_info['w'] = data['w_class']
            if 'dict_map_states' in data:
                additional_info['dict_map_states'] = data['dict_map_states']
            if 'num_states' in data:
                additional_info['num_states'] = data['num_states']

        return data[dict_name], additional_info

    def load_data_state(name):
        """ Load data and prepare dataloader for state data """
        additional_info = {}

        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            # num_types = int(data['dim_process'])

            if 'dim_process' in data:
                additional_info['num_types'] = data['dim_process']
            if 'num_marks' in data:
                additional_info['num_marks'] = data['num_marks']
            if 'dict_map_events' in data:
                additional_info['dict_map_events'] = data['dict_map_events']
            if 'pos_weight' in data:
                additional_info['pos_weight'] = data['pos_weight']
            if 'w_class' in data:
                additional_info['w'] = data['w_class']
            if 'dict_map_states' in data:
                additional_info['dict_map_states'] = data['dict_map_states']
            if 'num_states' in data:
                additional_info['num_states'] = data['num_states']
            if 'num_demos' in data:
                additional_info['num_demos'] = data['num_demos']

        return data, additional_info
    print('[Info] Loading train data...')
    train_data, additional_info = load_data_event(
        opt.data + 'train.pkl', 'train')
    print('[Info] Loading dev data...')
    valid_data, _ = load_data_event(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data_event(opt.data + 'test.pkl', 'test')

    if opt.per > 0:
        print(f'[info] {opt.per}% of data will be considered')
        train_data = train_data[:int(opt.per/100*len(train_data))]
        # test_data = test_data[:int(opt.per/100*len(test_data))]

    train_state = None
    test_state = None
    valid_state = None

    if opt.state or opt.sample_label:

        print('[Info] Loading train STATE...')
        train_state, new_additional_info = load_data_state(
            opt.data + 'train_state.pkl')
        print('[Info] Loading dev STATE...')
        valid_state, _ = load_data_state(opt.data + 'dev_state.pkl')
        print('[Info] Loading test STATE...')
        test_state, _ = load_data_state(opt.data + 'test_state.pkl')

        additional_info.update(new_additional_info)
        if opt.per > 0:
            print(f'[info] {opt.per}% of data will be considered')
            train_state['state'] = train_state['state'][:int(
                opt.per/100*len(train_state['state']))]
            # test_state['state'] = test_state['state'][:int(opt.per/100*len(test_state['state']))]
            if 'demo' in train_state.keys():
                train_state['demo'] = train_state['demo'][:int(
                    opt.per/100*len(train_state['demo']))]
                # test_state['demo'] = test_state['demo'][:int(opt.per/100*len(test_state['demo']))]

    state_args = {'have_label': opt.sample_label, 'have_demo': opt.demo}

    trainloader = get_dataloader3(train_data, data_state=train_state, bs=opt.batch_size, shuffle=True,
                                  data_label=opt.data_label, balanced=opt.balanced_batch, state_args=state_args)
    testloader = get_dataloader3(test_data, data_state=test_state, bs=opt.batch_size,
                                 shuffle=False, data_label=opt.data_label, balanced=False, state_args=state_args)
    validloader = get_dataloader3(valid_data, data_state=valid_state, bs=opt.batch_size,
                                  shuffle=False, data_label=opt.data_label, balanced=False, state_args=state_args)

    return trainloader, validloader, testloader, additional_info


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 1  # cumulative event log-likelihood
    total_time_se = 1  # cumulative time prediction squared-error
    total_event_rate = 1  # cumulative number of correct prediction
    total_num_event = 1  # number of total events
    total_num_pred = 1  # number of predictions, total=tqdm_len

    if opt.prof:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                opt.run_path),
            record_shapes=True,
            profile_memory=True,
            # with_stack=True
        )
        prof.start()

    log_loss = {'loss/event_decoder': 0, 'loss/pred_next_time': 0,
                'loss/pred_next_type': 0, 'loss/pred_label': 0}
    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):

        # if torch.isnan(model.TE.event_emb.weight).sum()>0:
        #     a=1

        batch = [x.to(opt.device) for x in batch]

        state_data = []
        state_label = None
        event_time, time_gap, event_type = batch[:3]
        if opt.state:
            state_time, state_value, state_mod = batch[3:6]
            state_data = batch[3:6]
        if opt.sample_label:
            state_time, state_value, state_mod = batch[3:6]
            state_label = batch[6]
        if opt.demo:
            state_data.append(batch[-1])

        enc_out = model(event_type, event_time, state_data=state_data)

        non_pad_mask = Utils.get_non_pad_mask(event_type).squeeze(2)

        # if torch.isnan(model.TE.event_emb.weight).sum()>0:
        #     a=1

        total_loss = []

        # CIF decoder
        if hasattr(model, 'event_decoder'):
            log_sum, integral_ = model.event_decoder(
                enc_out, event_time, event_type, non_pad_mask)
            loss_event_pp = (-torch.sum(log_sum - integral_)) * opt.w_event
            log_loss['loss/event_decoder'] += loss_event_pp.item()
            total_loss.append(loss_event_pp)

        # next type prediction
        if hasattr(model, 'pred_next_type'):
            next_type_loss, pred_num_event, _ = opt.type_loss(
                model.y_next_type, event_type, pred_loss_func)

            log_loss['loss/pred_next_type'] += next_type_loss.item()
            total_loss.append(next_type_loss)

        # next time prediction
        if hasattr(model, 'pred_next_time'):
            non_pad_mask = Utils.get_non_pad_mask(event_type).squeeze(-1)
            sse, sse_norm, sae = Utils.time_loss(
                model.y_next_time, event_time, non_pad_mask)  # sse, sse_norm, sae

            temp = sse*opt.w_time
            log_loss['loss/pred_next_time'] += temp.item()
            total_loss.append(temp)

        if hasattr(model, 'pred_label'):

            # state_label_red = state_label.sum(1).bool().int()[:,None,None] # [B,1,1]
            # state_label_loss,_ = Utils.state_label_loss(state_label_red,model.y_label, non_pad_mask)

            state_label_red = align(
                state_label[:, :, None], event_time, state_time)  # [B,L,1]
            state_label_loss, _ = Utils.state_label_loss(
                state_label_red, model.y_label, non_pad_mask, opt.label_loss_fun)

            temp = state_label_loss*opt.w_sample_label
            log_loss['loss/pred_label'] += temp.item()  # /event_time.shape[0]
            total_loss.append(temp)

        loss = sum(total_loss)

        """ forward """
        optimizer.zero_grad()

        loss.backward()
        # if torch.isnan(model.TE.event_emb.weight).sum()>0:
        #     a=1

        """ update parameters """

        optimizer.step()

        # if torch.isnan(model.TE.event_emb.weight).sum()>0:
        #     a=1

    rmse = np.sqrt(total_time_se / total_num_event)

    dict_metrics = {
        'NLL/#events': -total_event_ll / total_num_event,
        'acc': total_event_rate / total_num_pred,
        'RMSE': rmse,

    }

    dict_metrics.update(log_loss)

    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse, dict_metrics


def valid_epoch(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """
    example = {}
    model.eval()

    total_event_ll = 1  # cumulative event log-likelihood
    total_event_rate = 1  # cumulative number of correct prediction
    total_num_event = 1  # number of total events
    total_num_pred = 1  # number of predictions, total=tqdm_len

    total_time_sse = 1  # cumulative time prediction squared-error
    total_time_sae = 1  # cumulative time prediction squared-error
    total_time_sse_norm = 1  # cumulative time prediction squared-error

    total_label_state = 0  # cumulative time prediction squared-error

    time_gap_true = []
    time_gap_pred = []
    X_enc = []
    y_pred_list = []
    y_true_list = []
    y_score_list = []

    y_event_pred_list = []
    y_event_true_list = []
    y_event_score_list = []

    y_state_pred_list = []
    y_state_true_list = []
    y_state_score_list = []
    r_enc_list = []
    masks_list = []

    y_pred_stupid_list = []
    n_classes = model.n_marks

    dict_metrics = {}

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Testing)   ', leave=False):
            """ prepare data """

            batch = [x.to(opt.device) for x in batch]
            # batch = map(lambda x: x.to(opt.device), batch)

            state_data = []
            state_label = None
            # if opt.event_enc:
            event_time, time_gap, event_type = batch[:3]
            if opt.state:
                state_time, state_value, state_mod = batch[3:6]
                state_data = batch[3:6]
            if opt.sample_label:
                state_time, state_value, state_mod = batch[3:6]
                state_label = batch[6]
            if opt.demo:
                state_data.append(batch[-1])

            enc_out = model(event_type, event_time, state_data=state_data)

            non_pad_mask = Utils.get_non_pad_mask(event_type).squeeze(2)
            total_num_pred += non_pad_mask.sum().item()
            masks_list.append(
                non_pad_mask[:, 1:].flatten().bool().detach().cpu())  # [*, C]

            # CIF decoder
            if hasattr(model, 'event_decoder'):
                log_sum, integral_ = model.event_decoder(
                    enc_out, event_time, event_type, non_pad_mask)

                total_event_ll += torch.sum(log_sum - integral_)

                y_event_score_list.append(torch.flatten(
                    model.event_decoder.intens_at_evs, end_dim=1).detach().cpu())  # [*, n_cif]

            # next type prediction
            if hasattr(model, 'pred_next_type'):
                pred_loss, pred_num_event, (y_pred, y_true, y_score, masks) = opt.type_loss(
                    model.y_next_type, event_type, pred_loss_func)
                # total_loss.append(pred_loss)
                y_pred_list.append(torch.flatten(
                    y_pred, end_dim=1).detach().cpu())  # [*]
                y_true_list.append(torch.flatten(
                    y_true, end_dim=1).detach().cpu())  # [*]
                y_score_list.append(torch.flatten(
                    y_score, end_dim=1).detach().cpu())  # [*, C]

            # next time prediction
            if hasattr(model, 'pred_next_time'):
                # non_pad_mask = Utils.get_non_pad_mask(event_type).squeeze(-1)
                sse, sse_norm, sae = Utils.time_loss(
                    model.y_next_time, event_time, non_pad_mask)  # sse, sse_norm, sae
                # total_loss.append(sse*opt.w_time)
                total_time_sse += sse.item()  # cumulative time prediction squared-error
                total_time_sae += sae.item()  # cumulative time prediction squared-error
                # cumulative time prediction squared-error
                total_time_sse_norm += sse_norm.item()

            # label prediction
            if hasattr(model, 'pred_label') and (state_label is not None):

                # state_label_red = state_label.sum(1).bool().int()[:,None,None] # [B,1,1]
                # state_label_loss,(y_state_pred, y_state_true, y_state_score) = Utils.state_label_loss(state_label_red,model.y_label, non_pad_mask)

                state_label_red = align(
                    state_label[:, :, None], event_time, state_time)  # [B,L,1]
                # state_label_red = state_label.bool().int()[:,:,None] # [B,L,1]
                state_label_loss, (y_state_pred, y_state_true, y_state_score) = Utils.state_label_loss(
                    state_label_red, model.y_label, non_pad_mask, opt.label_loss_fun)

                # total_loss.append(state_label_loss*opt.w_sample_label)
                y_state_pred_list.append(torch.flatten(
                    y_state_pred).detach().cpu())  # [*]
                y_state_true_list.append(torch.flatten(
                    y_state_true).detach().cpu())  # [*]
                y_state_score_list.append(torch.flatten(
                    y_state_score).detach().cpu())  # [*] it is binary

    masks = torch.cat(masks_list)  # [*]

    # CIF decoder
    if hasattr(model, 'event_decoder'):

        dict_metrics.update({
            'CIF/LL-#events': total_event_ll.item() / total_num_pred,
            'CIF/NLL': -total_event_ll.item(),
            'CIF/#events': total_num_pred,
        })

    # next time prediction
    if hasattr(model, 'pred_next_time'):
        rmse = np.sqrt(total_time_sse / total_num_pred)
        msae = total_time_sae / total_num_pred
        rmse_norm = np.sqrt(total_time_sse_norm / total_num_pred)
        dict_metrics.update({
            'NextTime/RMSE': rmse,
            'NextTime/rmse_norm': rmse_norm,
            'NextTime/msae': msae,
        })

    # next type prediction
    if hasattr(model, 'pred_next_type'):

        if y_pred_list[-1].dim() == 2:  # multilabel or marked
            y_pred = (np.concatenate(y_pred_list)[masks, :])
            y_true = (np.concatenate(y_true_list)[masks, :])
            y_score = (np.concatenate(y_score_list)[masks, :])

            bad_labels = y_true.sum(0) == 0

            y_true = y_true[:, ~bad_labels]
            y_pred = y_pred[:, ~bad_labels]
            y_score = y_score[:, ~bad_labels]
            n_classes = y_true.shape[1]

            cm = metrics.multilabel_confusion_matrix(y_true, y_pred)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

            dict_metrics.update({

                'NextType(ML)/auc-ovo-weighted': metrics.roc_auc_score(y_true, y_score, multi_class='ovo', average='weighted'),
                # # 'NextType(ML)/auc-ovo-micro': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='micro'),
                # # 'NextType(ML)/auc-ovo-macro': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='macro'),

                'NextType(ML)/auc-PR-weighted': metrics.average_precision_score(y_true, y_score, average='weighted'),
                'NextType(ML)/f1-weighted': metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'NextType(ML)/precision-weighted': metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'NextType(ML)/recall-weighted': metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0),

                # 'NextType(ML)/f1-micro': metrics.f1_score(y_true, y_pred ,average='micro', zero_division=0),
                # 'NextType(ML)/f1-macro': metrics.f1_score(y_true, y_pred ,average='macro', zero_division=0),
                # 'NextType(ML)/f1-samples': metrics.f1_score(y_true, y_pred ,average='samples', zero_division=0),

                # 'NextType(ML)/acc': metrics.accuracy_score(y_true, y_pred),
                # 'NextType(ML)/hamming': metrics.hamming_loss(y_true, y_pred),

                # 'NextType(ML)/precision-micro': metrics.precision_score(y_true, y_pred ,average='micro', zero_division=0),
                # 'NextType(ML)/precision-macro': metrics.precision_score(y_true, y_pred ,average='macro', zero_division=0),
                # 'NextType(ML)/precision-samples': metrics.precision_score(y_true, y_pred ,average='samples', zero_division=0),

                # 'NextType(ML)/recall-micro': metrics.recall_score(y_true, y_pred ,average='micro', zero_division=0),
                # 'NextType(ML)/recall-macro': metrics.recall_score(y_true, y_pred ,average='macro', zero_division=0),
                # 'NextType(ML)/recall-samples': metrics.recall_score(y_true, y_pred ,average='samples', zero_division=0),

                # 'NextType(ML)/auc-ovo-macro': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='macro'),
                # 'NextType(ML)/auc-ovo-micro': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='micro'),
                # 'NextType(ML)/auc-PR-micro': metrics.average_precision_score(y_true, y_score ,average='micro'),
                # 'NextType(ML)/auc-PR-macro': metrics.average_precision_score(y_true, y_score ,average='macro'),

                # 'MultiLabel/AUROC-ovo-weighted': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='weighted'),
                # 'MultiLabel/AUPRC-weighted': metrics.average_precision_score(y_true, y_score ,average='weighted'),
                # 'MultiLabel/F1-weighted': metrics.f1_score(y_true, y_pred ,average='weighted', zero_division=0),

                # 'ConfMat': cm_display,

            })

            if hasattr(model, 'event_decoder') and model.event_decoder.n_cifs == n_classes:

                y_event_score = (np.concatenate(y_event_score_list)[masks, :])

                # y_event_score = nn.functional.normalize(y_event_score,p=1,dim=1)
                y_event_pred = (y_event_score > 0.5).astype(int)

                y_event_pred = y_event_pred[:, ~bad_labels]
                y_event_score = y_event_score[:, ~bad_labels]

                dict_metrics.update({

                    'NextType(ML)/auc-ovo-weighted-CIF': metrics.roc_auc_score(y_true, y_event_score, multi_class='ovo', average='weighted'),
                    'NextType(ML)/f1-weighted-CIF': metrics.f1_score(y_true,  y_event_pred, average='weighted', labels=torch.arange(n_classes)),
                })

        else:   # multiclass
            y_pred = (np.concatenate(y_pred_list)[masks])
            y_true = (np.concatenate(y_true_list)[masks])
            y_score = (np.concatenate(y_score_list)[masks, :])

            cm = metrics.confusion_matrix(y_true, y_pred)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

            if hasattr(model, 'event_decoder') and model.event_decoder.n_cifs == n_classes:

                y_event_score = (np.concatenate(y_event_score_list)[masks, :])
                y_event_score = y_event_score/y_event_score.sum(1)[:, None]
                y_event_pred = np.argmax(y_event_score, 1)

                if n_classes == 2:

                    dict_metrics.update({

                        'NextType(MC)/auc-ovo-weighted-CIF': metrics.roc_auc_score(y_true, y_score[:, 0], multi_class='ovo', average='weighted'),
                        'NextType(MC)/f1-weighted-CIF': metrics.f1_score(y_true,  y_event_pred, average='weighted', zero_division=0),
                    })
                else:
                    dict_metrics.update({

                        'NextType(MC)/auc-ovo-weighted-CIF': metrics.roc_auc_score(y_true, y_event_score, multi_class='ovo', average='weighted', labels=torch.arange(n_classes), ),
                        'NextType(MC)/f1-weighted-CIF': metrics.f1_score(y_true,  y_event_pred, average='weighted', labels=torch.arange(n_classes), zero_division=0),
                    })

            if n_classes == 2:
                dict_metrics.update({
                    # 'NextType(MC)/f1-micro': metrics.f1_score(y_true, y_pred, labels= torch.arange(n_classes) ,average='micro', zero_division=0),
                    # 'NextType(MC)/f1-macro': metrics.f1_score(y_true, y_pred,average='weighted'),
                    'NextType(MC)/f1-weighted': metrics.f1_score(y_true, y_pred, average='weighted'),
                    'NextType(MC)/precision-weighted': metrics.precision_score(y_true, y_pred),
                    'NextType(MC)/recall-weighted': metrics.recall_score(y_true, y_pred),

                    # 'NextType(MC)/auc-ovo-macro': metrics.roc_auc_score(y_true, y_score[:,0], multi_class='ovo',average='macro'),
                    'NextType(MC)/auc-weighted': metrics.roc_auc_score(y_true, y_score[:, 0], multi_class='ovo', average='weighted'),
                    # 'auc-ovo-weighted-stupid': metrics.roc_auc_score(y_true, y_pred_stupid, multi_class='ovo',average='weighted',labels= torch.arange(n_classes)),



                    'NextType(MC)/acc': metrics.accuracy_score(y_true, y_pred, normalize=True),

                    'ConfMat': cm_display,


                })
            else:
                dict_metrics.update({
                    # 'NextType(MC)/f1-micro': metrics.f1_score(y_true, y_pred, labels= torch.arange(n_classes) ,average='micro', zero_division=0),
                    'NextType(MC)/f1-macro': metrics.f1_score(y_true, y_pred, labels=torch.arange(n_classes), average='macro', zero_division=0),
                    'NextType(MC)/f1-weighted': metrics.f1_score(y_true, y_pred, labels=torch.arange(n_classes), average='weighted', zero_division=0),

                    # 'NextType(MC)/auc-ovo-macro': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='macro',labels= torch.arange(n_classes)),
                    'NextType(MC)/auc-weighted': metrics.roc_auc_score(y_true, y_score, multi_class='ovo', average='weighted', labels=torch.arange(n_classes)),
                    # 'auc-ovo-weighted-stupid': metrics.roc_auc_score(y_true, y_pred_stupid, multi_class='ovo',average='weighted',labels= torch.arange(n_classes)),



                    'NextType(MC)/acc': metrics.accuracy_score(y_true, y_pred, normalize=True),

                    'ConfMat': cm_display,


                })

    # label prediction
    if hasattr(model, 'pred_label'):

        # state_label_red = state_label.sum(1).bool().int()[:,None,None] # [B,1,1]
        # state_label_loss,(y_state_pred, y_state_true, y_state_score) = Utils.state_label_loss(state_label_red,model.y_label, non_pad_mask)
        # # total_loss.append(state_label_loss*opt.w_sample_label)
        # y_state_pred_list.append(y_state_pred) # [*]
        # y_state_true_list.append(y_state_true) # [*]
        # y_state_score_list.append(y_state_score) # [*] it is binary

        # y_state_pred = (np.concatenate(y_state_pred_list)) # [*]
        # y_state_true = (np.concatenate(y_state_true_list))
        # y_state_score = (np.concatenate(y_state_score_list))

        # y_state_pred = (np.concatenate(y_state_pred_list) [masks]) # [*]
        # y_state_true = (np.concatenate(y_state_true_list) [masks])
        # y_state_score = (np.concatenate(y_state_score_list) [masks])

        y_state_pred = (np.concatenate(y_state_pred_list))  # [*]
        y_state_true = (np.concatenate(y_state_true_list))
        y_state_score = (np.concatenate(y_state_score_list))
        # PR_CURVE = wandb.plot.pr_curve(y_state_true, y_state_score)

        pr, re, _ = metrics.precision_recall_curve(y_state_true, y_state_score)
        plt.figure()
        plt.plot(re, pr)
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        # PR_CURVE = px.line(x=re, y=pr, title='Precision-Recall Curve')
        dict_metrics.update({
            'pred_label/AUROC': metrics.roc_auc_score(y_state_true, y_state_score),
            'pred_label/AUPRC': metrics.average_precision_score(y_state_true, y_state_score),
            'pred_label/f1-binary': metrics.f1_score(y_state_true, y_state_pred, average='binary', zero_division=0),

            'pred_label/loss': total_label_state/total_num_event,
            'pred_label/recall-binary': metrics.recall_score(y_state_true, y_state_pred, average='binary', zero_division=0),
            'pred_label/precision-binary': metrics.precision_score(y_state_true, y_state_pred, average='binary', zero_division=0),

            'pred_label/ACC': metrics.accuracy_score(y_state_true, y_state_pred),

            'pred_label/PR_curve': plt,

            # 'pred_label/sum_r_enc':torch.concat(r_enc_list, dim=1).sum().item(),
            # 'pred_label/r_enc_zero': a,
        })

    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse, dict_metrics


def train(model, trainloader, validloader, testloader, optimizer, scheduler, pred_loss_func, opt, trial=None):
    """ Start training. """
    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE

    # PRE eval
    start = time.time()
    valid_event, valid_type, valid_time, dict_metrics_test = valid_epoch(
        model, testloader, pred_loss_func, opt)
    print('  - (PRE Testing)     loglikelihood: {ll: 8.5f}, '
          'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
          'elapse: {elapse:3.3f} min'
          .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

    # logging
    write_to_summary(dict_metrics_test, opt, i_epoch=0, prefix='Test-')

    best_metric = -100
    # Initialize the early stopping counter
    early_stopping_counter = 0

    # Set the maximum number of epochs without improvement
    max_epochs_without_improvement = opt.ES_pat

    dict_time = {}

    inter_Obj_val = 0
    best_test_metric = {}
    best_valid_metric = {}

    if opt.wandb_project in ['TEEDAM_unsupervised', 'TEEDAM_unsupervised_timeCat']:
        if opt.data_label == 'multilabel':
            best_test_metric.update({'NextType(ML)/auc-ovo-weighted': 0})
            best_valid_metric.update({'NextType(ML)/auc-ovo-weighted': 0})
        elif opt.data_label == 'multiclass':
            best_test_metric.update({'NextType(MC)/f1-weighted': 0})
            best_valid_metric.update({'NextType(MC)/f1-weighted': 0})
    elif opt.wandb_project == 'TEEDAM_supervised':
        best_test_metric.update({'pred_label/f1-binary': 0})
        best_valid_metric.update({'pred_label/f1-binary': 0})

    # if opt.sample_label:
    #     best_test_metric.update({'pred_label/f1-binary':0, 'pred_label/AUPRC':0, 'pred_label/AUROC':0})
    #     best_valid_metric.update({'pred_label/f1-binary':0, 'pred_label/AUPRC':0, 'pred_label/AUROC':0})
    # # elif opt.next_mark:

    # if opt.mod!='none':
    #     best_test_metric.update({'CIF/LL-#events':-100})
    #     best_valid_metric.update({'CIF/LL-#events':-100})
    # if opt.next_mark:
    #     if opt.data_label=='multilabel':
    #         best_test_metric.update({'NextType(ML)/auc-ovo-weighted':0, 'NextType(ML)/f1-weighted':0})
    #         best_valid_metric.update({'NextType(ML)/auc-ovo-weighted':0, 'NextType(ML)/f1-weighted':0})
    #     elif opt.data_label=='multiclass':
    #         best_test_metric.update({'NextType(MC)/auc-weighted':0, 'NextType(MC)/f1-weighted':0})
    #         best_valid_metric.update({'NextType(MC)/auc-weighted':0, 'NextType(MC)/f1-weighted':0})

    for epoch_i in tqdm(range(1, 1 + opt.epoch), leave=False):

        opt.i_epoch = epoch_i

        # ********************************************* Train Epoch *********************************************
        start = time.time()
        train_event, train_type, train_time, dict_metrics_train = train_epoch(
            model, trainloader, optimizer, pred_loss_func, opt)
        scheduler.step()

        wandb.log({'LR': optimizer.param_groups[0]['lr']}, step=epoch_i)
        dict_time.update({'Time/train_epoch': ((time.time() - start) / 60)})
        # for k, v in dict_metrics_train.items():
        #     opt.writer.add_scalar('Trainn/'+k, v, epoch_i)

        if opt.i_epoch % opt.log_freq == 0:

            # # Train eval *********************************************
            # train_event, train_type, train_time, dict_metrics_train2 = valid_epoch(
            #     model, trainloader, pred_loss_func, opt)
            # dict_metrics_train.update(dict_metrics_train2)
            # write_to_summary(dict_metrics_train, opt,
            #                  i_epoch=opt.i_epoch, prefix='Train-')

            # ********************************************* Valid Epoch *********************************************
            start = time.time()
            valid_event, valid_type, valid_time, dict_metrics_valid = valid_epoch(
                model, validloader, pred_loss_func, opt)

            dict_time.update(
                {'Time/valid_epoch': ((time.time() - start) / 60)})
            write_to_summary(dict_metrics_valid, opt,
                             i_epoch=opt.i_epoch, prefix='Valid-')

            # ********************************************* Test Epoch *********************************************
            if testloader is not None:

                test_event, test_type, test_time, dict_metrics_test = valid_epoch(
                    model, testloader, pred_loss_func, opt)

                write_to_summary(dict_metrics_test, opt,
                                 i_epoch=opt.i_epoch, prefix='Test-')

            write_to_summary(
                dict_time, opt, i_epoch=opt.i_epoch, prefix='time-')

            # objective value for HP Tuning
            if 'pred_label/f1-binary' in dict_metrics_test:
                inter_Obj_val = dict_metrics_test['pred_label/f1-binary']
            elif 'NextType(ML)/f1-weighted' in dict_metrics_test:
                inter_Obj_val = dict_metrics_test['NextType(ML)/f1-weighted']
            elif 'NextType(MC)/f1-weighted' in dict_metrics_test:
                inter_Obj_val = dict_metrics_test['NextType(MC)/f1-weighted']
            elif 'CIF/LL-#events' in dict_metrics_test:
                inter_Obj_val = dict_metrics_test['CIF/LL-#events']
            else:
                raise Exception("Sorry, no metrics for inter_Obj_val")

            opt.writer.add_scalar('Obj', inter_Obj_val,
                                  global_step=opt.i_epoch)
            if opt.wandb:
                wandb.log({'Obj': inter_Obj_val}, step=opt.i_epoch)

            # Early stopping
            flag = list()
            for k, v in best_valid_metric.items():
                if dict_metrics_valid[k] > v:
                    best_valid_metric[k] = dict_metrics_valid[k]
                    best_test_metric[k] = dict_metrics_test[k]
                    flag.append(k)
                    if opt.wandb:
                        # wandb.log({('Best-Test-'+k):v for k,v in best_test_metric.items()}, step=opt.i_epoch)
                        # wandb.log({('Best-Valid-'+k):v for k,v in best_valid_metric.items()}, step=opt.i_epoch)

                        dict_metrics_valid

                        wandb.log({('Best-Test-'+k1): v1 for k1,
                                  v1 in dict_metrics_test.items()}, step=opt.i_epoch)

                        wandb.log({('Best-Valid-'+k1): v1 for k1,
                                  v1 in dict_metrics_valid.items()}, step=opt.i_epoch)

                    # # # saving best torch model !!!
                    # if ('pred_label/f1-binary' in flag) or ('CIF/LL-#events' in flag) or ('NextType(ML)/auc-ovo-weighted' in flag)  or ('NextType(MC)/f1-weighted' in flag):

                    # opt.best_epoch = opt.i_epoch
                    wandb.log({'best_epoch': opt.i_epoch})
                    torch.save({
                        'epoch': opt.i_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'dict_metrics_test': dict_metrics_test,
                    }, opt.run_path+'/best_model.pkl')

            if inter_Obj_val > (best_metric+0.0001):
                # Save the model weights
                # torch.save(model.state_dict(), "best_model.pth")

                # Reset the early stopping counter
                early_stopping_counter = 0

                # Update the best metric
                best_metric = inter_Obj_val

            else:
                # Increment the early stopping counter
                early_stopping_counter += 1

                # Check if the early stopping counter has reached the maximum number of epochs without improvement
                if early_stopping_counter >= max_epochs_without_improvement:
                    print("Early stopping at epoch {}".format(opt.i_epoch))
                    if opt.wandb:
                        wandb.run.summary["max_obj_val"] = best_metric
                        wandb.run.summary["status"] = "stopped"
                        # wandb.finish(quiet=True)
                    break

            # Pruning
            if trial is not None:
                # raise optuna.TrialPruned()

             # Handle pruning based on the intermediate value.
                trial.report(inter_Obj_val, opt.i_epoch)
                if trial.should_prune():

                    opt.writer.add_hparams(opt.hparams2write, {
                                           'Objective': inter_Obj_val}, run_name='../'+opt.run_name)
                    return best_metric

    opt.writer.add_hparams(opt.hparams2write, {
                           'Objective': inter_Obj_val}, run_name='../'+opt.run_name)
    return best_metric


def options():
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument(
        '-data', default="C:/DATA/data/processed/physio2019_1d_HP_std/", required=False)
    parser.add_argument(
        '-data_label', choices=['multiclass', 'multilabel'], default='multilabel')

    parser.add_argument('-cuda', type=int,
                        choices=[0, 1], default=1, help='consider cuda?')

    parser.add_argument('-wandb', action='store_true',
                        dest='wandb', help='consider wandb?')
    parser.add_argument('-wandb_project', type=str,
                        default='TEDAM4', help='project name in wandb')

    parser.add_argument('-wandb_tag', type=str,
                        default='None', help='project name in wandb')

    parser.add_argument('-log_freq', type=int, default=1,
                        help='project name in wandb')

    parser.add_argument('-prof', action='store_true',
                        dest='prof', help='consider profiling?')

    parser.add_argument('-per', type=int, default=25,
                        help='percentage of dataset to be used for training')
    parser.add_argument('-unbalanced_batch', action='store_false',
                        dest='balanced_batch', help='not balanced mini batches?')

    parser.add_argument('-transfer_learning', default="",
                        help='specify run_name')

    parser.add_argument('-tl_tag', default="None",
                        help='specify tag')
    parser.add_argument(
        '-freeze', nargs='+', choices=['TE', 'DAM', ''], default='', help='specify run_name')

    parser.add_argument('-ES_pat', type=int, default=10,
                        help='max_epochs_without_improvement')

    # data handling

    parser.add_argument('-setting', type=str, choices=['sc', 'mc1', 'mc2', 'tl',
                        'rand', 'seft', 'raindrop'], default='', help='max_epochs_without_improvement')
    parser.add_argument('-test_center', type=str, default='',
                        help='max_epochs_without_improvement')
    parser.add_argument('-split', type=str, default='',
                        help='max_epochs_without_improvement')

    # logging args
    parser.add_argument('-log', type=str, default='log.txt')
    parser.add_argument('-user_prefix', type=str, default='SYN')
    parser.add_argument('-pos_alpha', type=float, default=1.0)

    # General Config
    parser.add_argument('-epoch', type=int, default=40)
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-lr', type=float, default=3e-4)
    parser.add_argument(
        '-lr_scheduler', choices=['StepLR', 'CosineAnnealingLR'], default='CosineAnnealingLR')
    parser.add_argument('-smooth', type=float, default=0.0)
    parser.add_argument('-weight_decay', type=float, default=1e-0)

    # ************************************** INPUT **************************************
    # Transformer Encoder Architecture
    parser.add_argument('-diag_offset', type=int,
                        default=1, help='offset for diagonal w=0,1,2 is diag=1,0,-1 for first j,j-1,j-2 to predict j+1')

    parser.add_argument('-event_enc', type=int,
                        choices=[0, 1], default=1, help='consider state?')

    parser.add_argument('-time_enc', type=str, choices=[
                        'sum', 'concat', 'none'], default='concat', help='specify time encoding')
    # TEE config
    parser.add_argument('--te_d_mark', type=int, default=8)
    parser.add_argument('--te_d_time', type=int, default=8)

    parser.add_argument('--te_d_rnn', type=int, default=256)
    parser.add_argument('--te_d_inner', type=int, default=16)
    parser.add_argument('--te_d_k', type=int, default=8)
    parser.add_argument('--te_d_v', type=int, default=8)
    parser.add_argument('--te_n_head', type=int, default=4)
    parser.add_argument('--te_n_layers', type=int, default=4)
    parser.add_argument('--te_dropout', type=float, default=0.1)

    # DAM config

    parser.add_argument('--dam_output_activation', type=str, default='relu')
    parser.add_argument('--dam_output_dims', type=int, default=16)
    parser.add_argument('--dam_n_phi_layers', type=int, default=3)
    parser.add_argument('--dam_phi_width', type=int, default=128)
    parser.add_argument('--dam_phi_dropout', type=float, default=0.2)
    parser.add_argument('--dam_n_psi_layers', type=int, default=2)
    parser.add_argument('--dam_psi_width', type=int, default=64)
    parser.add_argument('--dam_psi_latent_width', type=int, default=128)
    parser.add_argument('--dam_dot_prod_dim', type=int, default=64)
    parser.add_argument('--dam_n_heads', type=int, default=4)
    parser.add_argument('--dam_attn_dropout', type=float, default=0.1)
    parser.add_argument('--dam_latent_width', type=int, default=64)
    parser.add_argument('--dam_n_rho_layers', type=int, default=2)
    parser.add_argument('--dam_rho_width', type=int, default=128)
    parser.add_argument('--dam_rho_dropout', type=float, default=0.1)
    parser.add_argument('--dam_max_timescale', type=int, default=1000)
    parser.add_argument('--dam_n_positional_dims', type=int, default=16)
    parser.add_argument(
        '--dam_online', action='store_true',  dest='dam_online')
    # parser.add_argument('dam_num_mods',type=int, default=8)
    # parser.add_argument('dam_num_demos',type=int, default=8)

    # State Encoder Architecture
    parser.add_argument('-state', action='store_true',
                        dest='state', help='consider state?')
    parser.add_argument('-demo', action='store_true',
                        dest='demo', help='consider demo?')

    parser.add_argument('-num_states', type=int, default=1)
    # hyper paramters

    # noise
    parser.add_argument('-noise', action='store_true',
                        dest='noise', help='consider noise?')

    # Outputs

    # CIFs
    parser.add_argument(
        '-mod', type=str, choices=['single', 'mc', 'ml', 'none'], default='single', help='specify the mod')
    parser.add_argument('-int_dec', type=str, choices=[
                        'thp', 'sahp'], default='sahp', help='specify the inteesity decoder')
    parser.add_argument('-w_event', type=float, default=1)

    # marks
    parser.add_argument('-next_mark',  type=int,
                        choices=[0, 1], default=1, help='0: mark not detached, 1: mark detached')
    parser.add_argument('-w_class', action='store_true',
                        dest='w_class', help='consider w_class?')
    parser.add_argument('-w_pos', action='store_true',
                        dest='w_pos', help='consider w_pos?')
    # parser.add_argument('-mark_detach', action='store_true', dest='w_pos', help='consider w_pos?')
    parser.add_argument('-mark_detach',  type=int,
                        choices=[0, 1], default=0, help='0: mark not detached, 1: mark detached')

    # times
    parser.add_argument('-w_time', type=float, default=1.0)

    # final sample label
    # parser.add_argument('-sample_label', action='store_true', dest='sample_label', help='consider state?')
    parser.add_argument('-sample_label',  type=int,
                        choices=[0, 1, 2], default=0, help='2 for detach')
    parser.add_argument('-w_pos_label', type=float, default=1.0)
    parser.add_argument('-w_sample_label', type=float, default=10000.0)

    opt = parser.parse_args()

    temp = vars(opt)

    opt.hparams2write = {}
    for k, v in temp.items():
        if type(v) in [str, int, float, bool, torch.tensor]:
            opt.hparams2write[k] = v

    return opt


def config(opt, justLoad=False):
    if justLoad is False:

        # it is a new run
        t0 = datetime.datetime.strptime(
            "17-9-22--00-00-00", "%d-%m-%y--%H-%M-%S")
        t_now = datetime.datetime.now()
        t_diff = t_now-t0
        opt.date = time.strftime("%d-%m-%y--%H-%M-%S")
        # str(int(t_diff.seconds/10))
        opt.run_id = str(t_diff.days) + str(np.random.randint(1000, 10000))

        print(f"[Info] ### Point Process strategy: {opt.mod} ###")

        if 'data_so' in opt.data:
            opt.dataset = 'SO'
        if 'data_so_concat' in opt.data:
            opt.dataset = 'SO_CON'
        elif 'new_so' in opt.data:
            opt.dataset = 'NEW_SO'
        elif 'data_hawkes' in opt.data:
            opt.dataset = 'hawkes'
        if 'data_mimic' in opt.data:
            opt.dataset = 'MIMIC-II'
        elif 'MHP' in opt.data:
            opt.dataset = 'MHP'
        elif 'synthea' in opt.data:
            opt.dataset = 'Synthea'
        elif 'retweets_ml' in opt.data:
            opt.dataset = 'Retweets(ML)'
        elif 'retweets_mc' in opt.data:
            opt.dataset = 'Retweets(MC)'
        elif 'sahp_sim' in opt.data:
            opt.dataset = 'sim'
        elif 'p12' in opt.data:
            opt.dataset = 'P12'
        elif 'p19' in opt.data:
            opt.dataset = 'P19'

        if opt.setting == '':
            opt.str_config = '-'
            # Tensorboard integration
            opt.run_name = opt.user_prefix+str(opt.run_id)+opt.str_config

            if opt.split != '':
                opt.data = opt.data+'split'+opt.split+'/'
                print(opt.data)
            opt.run_path = opt.data + opt.run_name+'/'
        elif opt.setting in ['rand', 'seft']:
            opt.str_config = '-'+opt.setting
            # Tensorboard integration
            opt.run_name = opt.user_prefix+str(opt.run_id)
            opt.run_path = opt.data[:-1]+opt.str_config+'/' + opt.run_name+'/'
            # opt.dataset = opt.data
            opt.data = opt.data[:-1]+opt.str_config+'/'
        elif opt.setting in ['raindrop']:
            opt.str_config = '-'+opt.setting+'/split'+opt.split
            opt.run_name = opt.user_prefix+str(opt.run_id)
            opt.run_path = opt.data[:-1]+opt.str_config+'/' + opt.run_name+'/'

            opt.data = opt.data[:-1]+opt.str_config+'/'
        else:
            if opt.setting == 'mc2':
                opt.str_config = '-'+opt.setting+'-H'+opt.test_center
            elif opt.setting == 'tl':
                opt.str_config = '-'+'sc'+'-H'+opt.test_center+'/split'+opt.split
            else:
                opt.str_config = '-'+opt.setting+'-H'+opt.test_center+'/split'+opt.split
            # Tensorboard integration
            opt.run_name = opt.user_prefix+str(opt.run_id)
            opt.run_path = opt.data[:-1]+opt.str_config+'/' + opt.run_name+'/'

            # opt.dataset = opt.data
            opt.data = opt.data[:-1]+opt.str_config+'/'
        # create a foler for the run

        if os.path.exists(opt.run_path):
            # print(settings.load_model)
            shutil.rmtree(opt.run_path)
        os.makedirs(opt.run_path, exist_ok=True)

        with open(opt.run_path+'opt.pkl', 'wb') as f:
            pickle.dump(opt, f)

    if (opt.transfer_learning != '')*0:

        print('### ---------TRANSFER LEARNING----------->    '+opt.transfer_learning)

        # load opt file
        with open(opt.data + opt.transfer_learning+'/opt.pkl', 'rb') as f:
            opt_tl = pickle.load(f)

        print(f"""Available modules:

        opt.event_enc:              {opt_tl.event_enc}
        opt.state:                  {opt_tl.state}

        opt.next_mark:              {opt_tl.next_mark}
        opt.sample_label:           {opt_tl.sample_label}
        opt.mod:                    {opt_tl.mod}
        opt.int_dec:                {opt_tl.int_dec}

        
        
        """)
        print(opt_tl)

        opt.all_transfered_modules = []

        if opt_tl.state:
            opt.all_transfered_modules.append('DAM')
        if opt_tl.event_enc:
            opt.all_transfered_modules.append('TE')
        # if opt.next_mark:
        #     opt.all_transfered_modules.append('pred_next_type')

    else:
        opt_tl = opt

    opt.device = torch.device('cuda') if (
        torch.cuda.is_available() and opt.cuda) else torch.device('cpu')
    print(f"############################## CUDA {torch.cuda.is_available()}")

    # ###    What is the architecture?
    opt.INPUT = ''

    if opt.event_enc:
        opt.INPUT += 'TE'
    if opt.state:
        opt.INPUT += 'DAM'
    if opt.noise:
        opt.INPUT += 'NOISE'

    opt.OUTPUT = ''

    opt.OUTPUT += opt.mod
    opt.OUTPUT += '-mark' if opt.next_mark else ''
    opt.OUTPUT += '-label' if opt.sample_label else ''

    print(f'[Info] INPUT {opt.INPUT} --> {opt.OUTPUT}')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    # if justLoad is False:
    opt.trainloader, opt.validloader, opt.testloader, additional_info = prepare_dataloader(
        opt)

    if opt.mod == 'single' or opt.mod == 'none':
        opt.num_types = 1
        opt.num_types = additional_info['num_types']
    elif opt.mod == 'mc':
        opt.num_types = additional_info['num_types']
    elif opt.mod == 'ml':
        opt.num_types = additional_info['num_marks']

    if 'dict_map_events' in additional_info:
        opt.dict_map_events = additional_info['dict_map_events']
    if 'dict_map_states' in additional_info:
        opt.dict_map_states = additional_info['dict_map_states']

    if 'num_marks' in additional_info:
        opt.num_marks = additional_info['num_marks']
    else:
        opt.num_marks = opt.num_types
    if 'num_states' in additional_info:
        opt.num_states = additional_info['num_states']
    if 'pos_weight' in additional_info:
        opt.pos_weight = additional_info['pos_weight']

    if 'w' in additional_info:
        opt.w = additional_info['w']
    if 'num_marks' in additional_info:
        opt.num_marks = additional_info['num_marks']

    if 'num_demos' in additional_info:
        opt.num_demos = additional_info['num_demos']
    else:
        opt.num_demos = 0

    if opt.w_pos:

        opt.pos_weight = torch.tensor(opt.pos_weight, device=opt.device)
        opt.pos_weight = opt.pos_weight*opt.pos_alpha
        print('[Info] pos weigths:\n', opt.pos_weight)

    else:
        opt.pos_weight = torch.ones(
            opt.num_marks, device=opt.device)*opt.pos_alpha

    if opt.w_class:

        if opt.dataset == 'SO':
            opt.w = [0.0272, 0.0272, 0.0272, 0.0272, 0.0272, 0.0272, 0.0272, 0.0272, 0.0272,
                     0.0272, 0.0272, 0.0272, 0.0272, 0.0272, 0.0272, 0.0272, 0.0272, 0.0272,
                     0.0272, 0.0272, 0.1271, 0.3292]
            freq = np.array([3142,   874,   286, 22042,  2865,  2845,   996,   926, 11680,
                             685,   122,  1417,   402,   888,   135,    49,   151,   249,
                             167,    36,     9,     5])

            freq_per = freq/freq.sum()*100
            freq[freq_per < 1] = freq.sum()/100
            w_norm = 1.0 / np.sqrt(freq)

            opt.w = w_norm / w_norm.sum()

        opt.w = torch.tensor(opt.w, device=opt.device, dtype=torch.float32)
        print('[Info] w_class:\n', opt.w)

    else:
        opt.w = torch.ones(opt.num_marks, device=opt.device,
                           dtype=torch.float32)  # /opt.num_marks

    if opt.data_label == 'multilabel':
        opt.type_loss = Utils.type_loss_BCE
        opt.pred_loss_func = nn.BCEWithLogitsLoss(
            reduction='none', weight=opt.w, pos_weight=opt.pos_weight)
    elif opt.data_label == 'multiclass':
        opt.type_loss = Utils.type_loss_CE
        opt.pred_loss_func = nn.CrossEntropyLoss(
            ignore_index=-1, reduction='none', weight=opt.w)

    opt.label_loss_fun = nn.BCEWithLogitsLoss(
        reduction='none', pos_weight=torch.tensor(opt.w_pos_label, device=opt.device))

    opt.TE_config = {}
    if opt.event_enc:
        opt.TE_config['n_marks'] = opt.num_marks
        opt.TE_config['d_type_emb'] = opt.te_d_mark

        opt.TE_config['time_enc'] = opt.time_enc
        opt.TE_config['d_time'] = opt.te_d_time

        opt.TE_config['d_inner'] = opt.te_d_inner
        opt.TE_config['n_layers'] = opt.te_n_layers
        opt.TE_config['n_head'] = opt.te_n_head
        opt.TE_config['d_k'] = opt.te_d_k
        opt.TE_config['d_v'] = opt.te_d_v
        opt.TE_config['dropout'] = opt.te_dropout

    opt.DAM_config = {}
    if opt.state:

        # opt.DAM_config['output_activation'] = 'relu'
        # opt.DAM_config['output_dims'] = 4

        # # MLP encoder for combined values
        # opt.DAM_config['n_phi_layers'] = 3
        # opt.DAM_config['phi_width'] = 32
        # opt.DAM_config['phi_dropout'] = 0.2

        # # Cumulative Set Attention Layer
        # opt.DAM_config['n_psi_layers'] = 2
        # opt.DAM_config['psi_width'] = 16  # 16
        # opt.DAM_config['psi_latent_width'] = 32

        # opt.DAM_config['dot_prod_dim'] = 16  # 16
        # opt.DAM_config['n_heads'] = 2
        # opt.DAM_config['attn_dropout'] = 0.1
        # opt.DAM_config['latent_width'] = 16

        # opt.DAM_config['n_rho_layers'] = 2
        # opt.DAM_config['rho_width'] = 32  #
        # opt.DAM_config['rho_dropout'] = 0.1

        # opt.DAM_config['max_timescale'] = 1000
        # opt.DAM_config['n_positional_dims'] = 4
        # opt.DAM_config['num_mods'] = opt.num_states
        # opt.DAM_config['num_demos'] = opt.num_demos
        # opt.DAM_config['online'] = False

        # # complex 1
        # opt.DAM_config['output_activation'] = 'relu'
        # opt.DAM_config['output_dims'] = 8

        # # MLP encoder for combined values
        # opt.DAM_config['n_phi_layers'] = 3
        # opt.DAM_config['phi_width'] = 64
        # opt.DAM_config['phi_dropout'] = 0.2

        # # Cumulative Set Attention Layer
        # opt.DAM_config['n_psi_layers'] = 2
        # opt.DAM_config['psi_width'] = 32  # 16
        # opt.DAM_config['psi_latent_width'] = 64

        # opt.DAM_config['dot_prod_dim'] = 32  # 16
        # opt.DAM_config['n_heads'] = 4
        # opt.DAM_config['attn_dropout'] = 0.1
        # opt.DAM_config['latent_width'] = 32

        # opt.DAM_config['n_rho_layers'] = 2
        # opt.DAM_config['rho_width'] = 64  #
        # opt.DAM_config['rho_dropout'] = 0.1

        # opt.DAM_config['max_timescale'] = 1000
        # opt.DAM_config['n_positional_dims'] = 8
        # opt.DAM_config['num_mods'] = opt.num_states
        # opt.DAM_config['num_demos'] = opt.num_demos
        # opt.DAM_config['online'] = False

        # complex 2
        # opt.DAM_config['output_activation'] = 'relu'
        # opt.DAM_config['output_dims'] = 16

        # # MLP encoder for combined values
        # opt.DAM_config['n_phi_layers'] = 3
        # opt.DAM_config['phi_width'] = 128
        # opt.DAM_config['phi_dropout'] = 0.2

        # # Cumulative Set Attention Layer
        # opt.DAM_config['n_psi_layers'] = 2
        # opt.DAM_config['psi_width'] = 64  # 16
        # opt.DAM_config['psi_latent_width'] = 128

        # opt.DAM_config['dot_prod_dim'] = 64  # 16
        # opt.DAM_config['n_heads'] = 4
        # opt.DAM_config['attn_dropout'] = 0.1
        # opt.DAM_config['latent_width'] = 64

        # opt.DAM_config['n_rho_layers'] = 2
        # opt.DAM_config['rho_width'] = 128  #
        # opt.DAM_config['rho_dropout'] = 0.1

        # opt.DAM_config['max_timescale'] = 1000
        # opt.DAM_config['n_positional_dims'] = 16
        # opt.DAM_config['num_mods'] = opt.num_states
        # opt.DAM_config['num_demos'] = opt.num_demos
        # opt.DAM_config['online'] = False

        # NEW
        opt.DAM_config['output_activation'] = opt.dam_output_activation
        opt.DAM_config['output_dims'] = opt.dam_output_dims

        # MLP encoder for combined values
        opt.DAM_config['n_phi_layers'] = opt.dam_n_phi_layers
        opt.DAM_config['phi_width'] = opt.dam_phi_width
        opt.DAM_config['phi_dropout'] = opt.dam_phi_dropout

        # Cumulative Set Attention Layer
        opt.DAM_config['n_psi_layers'] = opt.dam_n_psi_layers
        opt.DAM_config['psi_width'] = opt.dam_psi_width
        opt.DAM_config['psi_latent_width'] = opt.dam_psi_latent_width

        opt.DAM_config['dot_prod_dim'] = opt.dam_dot_prod_dim
        opt.DAM_config['n_heads'] = opt.dam_n_heads
        opt.DAM_config['attn_dropout'] = opt.dam_attn_dropout
        opt.DAM_config['latent_width'] = opt.dam_latent_width

        opt.DAM_config['n_rho_layers'] = opt.dam_n_rho_layers
        opt.DAM_config['rho_width'] = opt.dam_rho_width
        opt.DAM_config['rho_dropout'] = opt.dam_rho_dropout

        opt.DAM_config['max_timescale'] = opt.dam_max_timescale
        opt.DAM_config['n_positional_dims'] = opt.dam_n_positional_dims
        opt.DAM_config['online'] = opt.dam_online

        opt.DAM_config['num_mods'] = opt.num_states
        opt.DAM_config['num_demos'] = opt.num_demos

    opt.NOISE_config = {}
    if opt.noise:
        opt.NOISE_config['noise_size'] = 32*0

    opt.demo_config = {}
    if opt.demo:
        opt.demo_config['num_demos'] = additional_info['num_demos']
        opt.demo_config['d_demo'] = 4

    opt.CIF_config = {}
    if opt.mod != 'none':
        opt.CIF_config['mod'] = opt.mod
        opt.CIF_config['type'] = opt.int_dec

        if opt.CIF_config['mod'] == 'single':
            opt.CIF_config['n_cifs'] = 1
        else:
            opt.CIF_config['n_cifs'] = opt.num_marks

    opt.next_type_config = {}
    if opt.next_mark:
        opt.next_type_config['n_marks'] = opt.num_marks
        opt.next_type_config['mark_detach'] = opt.mark_detach
    opt.next_time_config = True

    opt.label_config = {}
    if opt.sample_label:
        opt.label_config['sample_detach'] = 1 if (opt.sample_label == 2) else 0

    return opt


def process_hparams(trial, opt):

    # argv=['Main.py']
    # for hparam in hparams:
    #     # argv.extend([k,str(v)])
    #     if hparam != '':
    #         argv.append(str(hparam))

    # print(argv)
    print('############################################ HP')

    # opt.w_pos_label = trial.suggest_float('w_pos_label',0.2, 5,step=0.1)

    # opt.weight_decay = trial.suggest_float('weight_decay',1e-4, 1,log=True)
    # opt.lr = trial.suggest_categorical('lr', [1e-4,1e-3,1e-2])
    # opt.lr = trial.suggest_categorical('lr', [0.03,0.0003])

    # opt.weight_decay = trial.suggest_categorical('weight_decay', [1e-3,1e-2,1e-1,1])
    # opt.w_pos_label = trial.suggest_categorical('w_pos_label', [0.1, 0.3,0.5,0.7,1,2])

    opt.lr = trial.suggest_categorical('lr', [0.01, 0.001, 0.0003, 0.0001])

    opt.batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    if opt.dataset == 'synthea_full':
        # opt.lr = trial.suggest_float('lr',0.0003, 0.03,log=True)
        opt.lr = trial.suggest_categorical('lr', [0.001, 0.003, 0.01])

        opt.batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        opt.te_d_mark = trial.suggest_categorical(
            'te_d_mark', [16, 32, 64, 128])
    if opt.dataset == 'SO':
        # opt.lr = trial.suggest_float('lr',0.0003, 0.03,log=True)
        opt.lr = trial.suggest_categorical('lr', [0.0001, 0.0003, 0.001])

        opt.batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])

    if opt.event_enc:
        # # # opt.TE_config['n_marks'] = opt.num_marks
        # opt.TE_config['d_type_emb'] = trial.suggest_categorical('d_type_emb', [8,16])

        # # opt.TE_config['time_enc'] = trial.suggest_categorical('time_enc', ['sum','concat','none'])
        # opt.TE_config['d_time'] = trial.suggest_categorical('d_time', [8,16])

        # opt.TE_config['d_inner'] = trial.suggest_categorical('d_inner', [8,16])
        # opt.TE_config['n_layers'] = trial.suggest_categorical('n_layers', [2,4,8])
        # opt.TE_config['n_head'] = trial.suggest_categorical('n_head', [2,4,8])
        # opt.TE_config['d_k'] = trial.suggest_categorical('d_k', [8])
        # opt.TE_config['d_v'] = trial.suggest_categorical('d_v', [8])
        # opt.TE_config['dropout'] = trial.suggest_categorical('dropout', [8,16])
        aaa = 1

    if opt.state:

        # opt.DAM_config['output_activation'] = trial.suggest_categorical('output_activation', [8,16])
        # opt.DAM_config['output_dims'] = trial.suggest_categorical('output_dims', [8,16])

        #             # MLP encoder for combined values
        # opt.DAM_config['n_phi_layers'] = trial.suggest_categorical('n_phi_layers', [8,16])
        # opt.DAM_config['phi_width'] = trial.suggest_categorical('phi_width', [8,16])
        # opt.DAM_config['phi_dropout'] = trial.suggest_categorical('phi_dropout', [8,16])

        #             # Cumulative Set Attention Layer
        # opt.DAM_config['n_psi_layers'] = trial.suggest_categorical('n_psi_layers', [8,16])
        # opt.DAM_config['psi_width'] = trial.suggest_categorical('psi_width', [8,16])
        # opt.DAM_config['psi_latent_width'] = trial.suggest_categorical('psi_latent_width', [8,16])

        # opt.DAM_config['dot_prod_dim'] = 16 # trial.suggest_categorical('dot_prod_dim', [16,32,64])
        # opt.DAM_config['n_heads'] = 2 #trial.suggest_categorical('n_heads', [2,4])
        # opt.DAM_config['attn_dropout'] = trial.suggest_categorical('attn_dropout', [8,16])
        # opt.DAM_config['latent_width'] = 16 #trial.suggest_categorical('latent_width', [16,128])

        # opt.DAM_config['n_rho_layers'] = trial.suggest_categorical('n_rho_layers', [8,16])
        # opt.DAM_config['rho_width'] = 256 #trial.suggest_categorical('rho_width', [32,256])
        # opt.DAM_config['rho_dropout'] = trial.suggest_categorical('rho_dropout', [8,16])

        # opt.DAM_config['max_timescale'] = trial.suggest_categorical('max_timescale', [8,16])
        # opt.DAM_config['n_positional_dims'] = trial.suggest_categorical('n_positional_dims', [8,16])
        # # # opt.DAM_config ['num_mods'] = opt.num_states
        # # # opt.DAM_config ['num_demos'] = opt.num_demos
        # # # opt.DAM_config ['online'] = False

        aa = 1

    config_hparams = dict(trial.params)
    config_hparams["trial.number"] = trial.number

    return opt, config_hparams


def load_module(model, checkpoint, modules, to_freeze=True):

    for module in modules:

        b = [x for x in checkpoint['model_state_dict'].keys()
             if x.startswith(module)]
        od = OrderedDict()
        for k in b:
            od[k[(len(module)+1):]] = checkpoint['model_state_dict'][k]

        # model.encoder.load_state_dict(od)
        getattr(model, module).load_state_dict(od, strict=False)
        for para in getattr(model, module).parameters():
            para.requires_grad = not to_freeze

    return


def main(trial=None):
    """ Main function. """

    print(sys.argv)

    opt = options()  # if run from command line it process sys.argv

    opt = config(opt)

    # torch.manual_seed(int(opt.run_id))
    torch.manual_seed(42)

    if isinstance(trial, optuna.trial._trial.Trial):
        print('OPTUNA!')
        opt, config_hparams = process_hparams(trial, opt)
        opt.hparams = config_hparams
        if opt.wandb:
            wandb.login()
            # wandb.tensorboard.patch(root_logdir=opt.run_path, pytorch=True)
            # sync_tensorboard=True,
            wandb.init(config=opt,
                       project=opt.wandb_project,
                       entity="hokarami",
                       group=opt.user_prefix,
                       name=opt.run_name,
                       reinit=True,
                       tags=[opt.wandb_tag],
                       # settings=wandb.Settings(start_method="fork")
                       )
            # wandb.config.update(opt.TE_config)
            # wandb.config.update(opt.DAMconfig)
            # opt.wandb_dir = wandb.run.dir
            # shutil.copy(opt.run_path+'opt.pkl',wandb.run.dir+'/opt.pkl')

        opt.writer = SummaryWriter(log_dir=opt.run_path)
        print(
            f"[info] tensorboard integration:\ntensorboard --logdir '{opt.data}'\n")
    else:

        if opt.wandb:
            wandb.login()
            # wandb.tensorboard.patch(root_logdir=opt.run_path, pytorch=True)
            # sync_tensorboard=True,
            wandb.init(config=opt, project=opt.wandb_project,
                       entity="hokarami", name=opt.run_name, tags=[opt.wandb_tag],)
            # wandb.config.update(opt.TE_config)
            # wandb.config.update(opt.DAMconfig)
            # opt.wandb_dir = wandb.run.dir

            # shutil.copy(opt.run_path+'opt.pkl',wandb.run.dir+'/opt.pkl')

        opt.writer = SummaryWriter(log_dir=opt.run_path)
        print(
            f"[info] tensorboard integration:\ntensorboard --logdir '{opt.data}'\n")

    opt.label_loss_fun = nn.BCEWithLogitsLoss(
        reduction='none', pos_weight=torch.tensor(opt.w_pos_label, device=opt.device))

    """ prepare model """
    model = ATHP(
        n_marks=opt.num_marks,
        TE_config=opt.TE_config,
        DAM_config=opt.DAM_config,
        NOISE_config=opt.NOISE_config,

        CIF_config=opt.CIF_config,
        next_time_config=opt.next_time_config,
        next_type_config=opt.next_type_config,
        label_config=opt.label_config,

        demo_config=opt.demo_config,

        device=opt.device,
        diag_offset=opt.diag_offset

    )
    model.to(opt.device)

    if opt.transfer_learning != '':

        # files = os.listdir(opt.data+TL_run)

        # # find last epoch
        # all_epochs=[]
        # for file in files:
        #     if 'model_ep' in file:
        #         all_epochs.append(int(re.findall(r'\d+', file)[0]))
        # if len(all_epochs)==0:
        #     raise Exception('No models were found for transfer learning!!!')
        # MODEL_PATH = opt.data+TL_run+'/model_ep'+str(max(all_epochs))+'.pkl'

        api = wandb.Api()
        runs = api.runs("hokarami/TEEDAM_unsupervised")
        df_filt = dl_runs(runs, selected_tag=opt.tl_tag)

        df_config = pd.DataFrame(
            [{k: v for k, v in x.items()} for x in df_filt.config])
        df_summary = pd.DataFrame(
            [{k: v for k, v in x.items()} for x in df_filt.summary])
        df_path = df_filt.path.apply(lambda x: '/'.join(x))
        df = pd.concat([df_config, df_summary, df_path], axis=1)

        q = (df['dataset'] == opt.dataset) & (df['setting'] == opt.setting) & (df['INPUT'] == opt.INPUT) & (
            df['diag_offset'] == opt.diag_offset) & (df['test_center'] == opt.test_center) & (df['split'] == opt.split)

        if q.sum() != 1:
            print('### COULD NOT FIND UNIQUE RUN FOR TL')
            raise "COULD NOT FIND UNIQUE RUN FOR TL"
        else:
            # sth like 'hokarami/TEEDAM_unsupervised/8bhh70yz'
            run_path = df.loc[q]['path'].values[0]

            run = api.run(run_path)
            for file in run.files():
                file.download(replace=True, root=f'./local/{run_path}/')
            checkpoint = torch.load(f'./local/{run_path}/best_model.pkl')

        # MODEL_PATH = opt.data + opt.transfer_learning+'/best_model.pkl'

        # checkpoint = torch.load(MODEL_PATH)

        # for para in model.parameters():
        #     para.requires_grad = False
        opt.all_transfered_modules = opt.freeze
        load_module(model, checkpoint,
                    modules=opt.all_transfered_modules, to_freeze=True)
        print('### [info] all transfered modules: ',
              opt.all_transfered_modules)

        # load_module(model, checkpoint, modules= opt.freezed_modules, to_freeze=False)
        # print('### [info] unfreezed modules: ',opt.freezed_modules)

        # model.A_reg.requires_grad=True

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=opt.weight_decay)
    # optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
    #                        opt.lr,momentum=0.01, weight_decay=opt.weight_decay)

    if opt.lr_scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
    elif opt.lr_scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=0.00001)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=10, T_mult=1, eta_min=0.00001)
    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    wandb.log({'num_params': num_params})
    """ train the model """

    max_obj_val = 0
    if opt.state:

        max_obj_val = train(model, opt.trainloader, opt.validloader,
                            opt.testloader, optimizer, scheduler, opt.pred_loss_func, opt, trial)

    else:

        max_obj_val = train(model, opt.trainloader, opt.validloader,
                            opt.testloader, optimizer, scheduler, opt.pred_loss_func, opt, trial)

    shutil.copy(opt.run_path+'opt.pkl', wandb.run.dir+'/opt.pkl')
    shutil.copy(opt.run_path+'best_model.pkl', wandb.run.dir+'/best_model.pkl')

    if isinstance(trial, optuna.trial._trial.Trial) and trial.should_prune():
        if opt.wandb:
            wandb.run.summary["state"] = "pruned"

            wandb.save('opt.pkl')
            wandb.save('best_model.pkl')
            wandb.finish(quiet=True)
        raise optuna.exceptions.TrialPruned()

    if opt.wandb:

        # report the final validation accuracy to wandb
        wandb.run.summary["max_obj_val"] = max_obj_val
        wandb.run.summary["status"] = "completed"

        wandb.save('opt.pkl')
        wandb.save('best_model.pkl')
        wandb.finish(quiet=True)

    return max_obj_val


if __name__ == '__main__':
    
    # setup wandb
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_KEY"))


    main()


def valid_epoch_tsne(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """
    example = {}
    model.eval()

    out = dict()

    total_event_ll = 1  # cumulative event log-likelihood
    total_event_rate = 1  # cumulative number of correct prediction
    total_num_event = 1  # number of total events
    total_num_pred = 1  # number of predictions, total=tqdm_len

    total_time_sse = 1  # cumulative time prediction squared-error
    total_time_sae = 1  # cumulative time prediction squared-error
    total_time_sse_norm = 1  # cumulative time prediction squared-error

    total_label_state = 0  # cumulative time prediction squared-error

    event_type_list = []
    event_time_list = []
    next_event_type_list = []

    non_pad_mask_list = []

    time_gap_true = []
    time_gap_pred = []
    X_enc = []
    y_pred_list = []
    y_true_list = []
    y_score_list = []

    y_event_pred_list = []
    y_event_true_list = []
    y_event_score_list = []

    y_state_pred_list = []
    y_state_true_list = []
    y_state_score_list = []
    r_enc_list = []
    r_enc_list2 = []
    masks_list = []

    state_mod_list = []
    state_time_list = []
    state_value_list = []

    y_pred_stupid_list = []
    n_classes = model.n_marks

    list_intens_at_samples = []
    list_taus = []
    list_true_intens_at_evs = []
    list_log_sum = []
    list_integral_ = []

    list_TE_att = []
    list_DAM_att = []

    dict_metrics = {}

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Testing)   ', leave=False):
            """ prepare data """

            batch = [x.to(opt.device) for x in batch]
            # batch = map(lambda x: x.to(opt.device), batch)

            state_data = []
            state_label = None
            # if opt.event_enc:
            event_time, time_gap, event_type = batch[:3]
            if opt.state:
                state_time, state_value, state_mod = batch[3:6]
                state_mod_list.append(state_mod.detach().cpu())
                state_time_list.append(state_time.detach().cpu())
                state_value_list.append(state_value.detach().cpu())

                state_data = batch[3:6]
            if opt.sample_label:
                state_time, state_value, state_mod = batch[3:6]
                state_label = batch[6]
            if opt.demo:
                state_data.append(batch[-1])

            non_pad_mask = Utils.get_non_pad_mask(event_type).squeeze(2)

            enc_out = model(event_type, event_time, state_data=state_data)

            a = (non_pad_mask.sum(1)-1).int().tolist()
            r = enc_out[[i for i in range(enc_out.shape[0])], a, :]
            r_enc_list.append(r.detach().cpu())

            # r_enc_list.append(enc_out.detach().cpu()[:,-1,:])
            # r_enc_list2.append(enc_out.detach().cpu()[:,-2,:])
            # next_list2.append(event_type.detach().cpu()[:])
            event_type_list.append(event_type.detach().cpu())
            if (event_type.sum(-1).sum(-1) == 0).any() == 0:
                print("TOO BAD")
            event_time_list.append(event_time.detach().cpu())

            non_pad_mask_list.append(non_pad_mask)

            total_num_pred += non_pad_mask.sum().item()
            masks_list.append(
                non_pad_mask[:, 1:].flatten().bool().detach().cpu())  # [*, C]

            # CIF decoder
            if hasattr(model, 'event_decoder'):
                log_sum, integral_ = model.event_decoder(
                    enc_out, event_time, event_type, non_pad_mask)

                list_intens_at_samples.append(
                    model.event_decoder.intens_at_samples.detach().cpu())
                list_taus.append(model.event_decoder.taus.detach().cpu())
                list_true_intens_at_evs.append(
                    model.event_decoder.true_intens_at_evs.detach().cpu())
                list_log_sum.append(log_sum.detach().cpu())
                list_integral_.append(integral_.detach().cpu())

                total_event_ll += torch.sum(log_sum - integral_)

                y_event_score_list.append(torch.flatten(
                    model.event_decoder.intens_at_evs, end_dim=1).detach().cpu())  # [*, n_cif]

            if hasattr(model, 'TE'):
                list_TE_att.append(model.TE.self_attn)
            if hasattr(model, 'DAM'):
                list_DAM_att.append(model.DAM.att)

            # next type prediction
            if hasattr(model, 'pred_next_type'):
                pred_loss, pred_num_event, (y_pred, y_true, y_score, masks) = opt.type_loss(
                    model.y_next_type, event_type, pred_loss_func)
                # total_loss.append(pred_loss)
                y_pred_list.append(torch.flatten(
                    y_pred, end_dim=1).detach().cpu())  # [*]
                y_true_list.append(torch.flatten(
                    y_true, end_dim=1).detach().cpu())  # [*]
                y_score_list.append(torch.flatten(
                    y_score, end_dim=1).detach().cpu())  # [*, C]

                next_event_type_list.append(y_pred.detach().cpu())

            # next time prediction
            if hasattr(model, 'pred_next_time'):
                # non_pad_mask = Utils.get_non_pad_mask(event_type).squeeze(-1)
                sse, sse_norm, sae = Utils.time_loss(
                    model.y_next_time, event_time, non_pad_mask)  # sse, sse_norm, sae
                # total_loss.append(sse*opt.w_time)
                total_time_sse += sse.item()  # cumulative time prediction squared-error
                total_time_sae += sae.item()  # cumulative time prediction squared-error
                # cumulative time prediction squared-error
                total_time_sse_norm += sse_norm.item()

            # label prediction
            if hasattr(model, 'pred_label') and (state_label is not None):

                # state_label_red = state_label.sum(1).bool().int()[:,None,None] # [B,1,1]
                # state_label_loss,(y_state_pred, y_state_true, y_state_score) = Utils.state_label_loss(state_label_red,model.y_label, non_pad_mask)

                state_label_red = align(
                    state_label[:, :, None], event_time, state_time)  # [B,L,1]
                # state_label_red = state_label.bool().int()[:,:,None] # [B,L,1]
                state_label_loss, (y_state_pred, y_state_true, y_state_score) = Utils.state_label_loss(
                    state_label_red, model.y_label, non_pad_mask, opt.label_loss_fun)

                # total_loss.append(state_label_loss*opt.w_sample_label)
                y_state_pred_list.append(torch.flatten(
                    y_state_pred).detach().cpu())  # [*]
                y_state_true_list.append(torch.flatten(
                    y_state_true).detach().cpu())  # [*]
                y_state_score_list.append(torch.flatten(
                    y_state_score).detach().cpu())  # [*] it is binary

    masks = torch.cat(masks_list).detach().cpu()  # [*]

    out['non_pad_mask_list'] = non_pad_mask_list

    out['event_type_list'] = event_type_list
    out['event_time_list'] = event_time_list
    out['next_event_type_list'] = next_event_type_list

    out['list_intens_at_samples'] = list_intens_at_samples
    out['list_taus'] = list_taus
    out['list_true_intens_at_evs'] = list_true_intens_at_evs
    out['list_log_sum'] = list_log_sum
    out['list_integral_'] = list_integral_

    out['state_mod_list'] = state_mod_list
    out['state_time_list'] = state_time_list
    out['state_value_list'] = state_value_list

    out['r_enc_list'] = r_enc_list

    out['list_TE_att'] = list_TE_att
    out['list_DAM_att'] = list_DAM_att

    # CIF decoder
    if hasattr(model, 'event_decoder'):

        dict_metrics.update({
            'CIF/LL-#events': total_event_ll.item() / total_num_pred,
            'CIF/NLL': -total_event_ll.item(),
            'CIF/#events': total_num_pred,
        })

    # next time prediction
    if hasattr(model, 'pred_next_time'):
        rmse = np.sqrt(total_time_sse / total_num_pred)
        msae = total_time_sae / total_num_pred
        rmse_norm = np.sqrt(total_time_sse_norm / total_num_pred)
        dict_metrics.update({
            'NextTime/RMSE': rmse,
            'NextTime/rmse_norm': rmse_norm,
            'NextTime/msae': msae,
        })

    # next type prediction
    if hasattr(model, 'pred_next_type'):

        if y_pred_list[-1].dim() == 2:  # multilabel or marked
            y_pred = (np.concatenate(y_pred_list)[masks, :])
            y_true = (np.concatenate(y_true_list)[masks, :])
            y_score = (np.concatenate(y_score_list)[masks, :])

            bad_labels = y_true.sum(0) == 0

            y_true = y_true[:, ~bad_labels]
            y_pred = y_pred[:, ~bad_labels]
            y_score = y_score[:, ~bad_labels]
            n_classes = y_true.shape[1]

            cm = metrics.multilabel_confusion_matrix(y_true, y_pred)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

            dict_metrics.update({

                'NextType(ML)/auc-ovo-weighted': metrics.roc_auc_score(y_true, y_score, multi_class='ovo', average='weighted'),
                # # 'NextType(ML)/auc-ovo-micro': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='micro'),
                # # 'NextType(ML)/auc-ovo-macro': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='macro'),

                'NextType(ML)/auc-PR-weighted': metrics.average_precision_score(y_true, y_score, average='weighted'),
                'NextType(ML)/f1-weighted': metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'NextType(ML)/precision-weighted': metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'NextType(ML)/recall-weighted': metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0),

                # 'NextType(ML)/f1-micro': metrics.f1_score(y_true, y_pred ,average='micro', zero_division=0),
                # 'NextType(ML)/f1-macro': metrics.f1_score(y_true, y_pred ,average='macro', zero_division=0),
                # 'NextType(ML)/f1-samples': metrics.f1_score(y_true, y_pred ,average='samples', zero_division=0),

                # 'NextType(ML)/acc': metrics.accuracy_score(y_true, y_pred),
                # 'NextType(ML)/hamming': metrics.hamming_loss(y_true, y_pred),

                # 'NextType(ML)/precision-micro': metrics.precision_score(y_true, y_pred ,average='micro', zero_division=0),
                # 'NextType(ML)/precision-macro': metrics.precision_score(y_true, y_pred ,average='macro', zero_division=0),
                # 'NextType(ML)/precision-samples': metrics.precision_score(y_true, y_pred ,average='samples', zero_division=0),

                # 'NextType(ML)/recall-micro': metrics.recall_score(y_true, y_pred ,average='micro', zero_division=0),
                # 'NextType(ML)/recall-macro': metrics.recall_score(y_true, y_pred ,average='macro', zero_division=0),
                # 'NextType(ML)/recall-samples': metrics.recall_score(y_true, y_pred ,average='samples', zero_division=0),

                # 'NextType(ML)/auc-ovo-macro': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='macro'),
                # 'NextType(ML)/auc-ovo-micro': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='micro'),
                # 'NextType(ML)/auc-PR-micro': metrics.average_precision_score(y_true, y_score ,average='micro'),
                # 'NextType(ML)/auc-PR-macro': metrics.average_precision_score(y_true, y_score ,average='macro'),

                # 'MultiLabel/AUROC-ovo-weighted': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='weighted'),
                # 'MultiLabel/AUPRC-weighted': metrics.average_precision_score(y_true, y_score ,average='weighted'),
                # 'MultiLabel/F1-weighted': metrics.f1_score(y_true, y_pred ,average='weighted', zero_division=0),

                # 'ConfMat': cm_display,

            })

            if hasattr(model, 'event_decoder') and model.event_decoder.n_cifs == n_classes:

                y_event_score = (np.concatenate(y_event_score_list)[masks, :])

                # y_event_score = nn.functional.normalize(y_event_score,p=1,dim=1)
                y_event_pred = (y_event_score > 0.5).astype(int)

                y_event_pred = y_event_pred[:, ~bad_labels]
                y_event_score = y_event_score[:, ~bad_labels]

                dict_metrics.update({

                    'NextType(ML)/auc-ovo-weighted-CIF': metrics.roc_auc_score(y_true, y_event_score, multi_class='ovo', average='weighted'),
                    'NextType(ML)/f1-weighted-CIF': metrics.f1_score(y_true,  y_event_pred, average='weighted', labels=torch.arange(n_classes)),
                })

        else:   # multiclass
            y_pred = (np.concatenate(y_pred_list)[masks])
            y_true = (np.concatenate(y_true_list)[masks])
            y_score = (np.concatenate(y_score_list)[masks, :])

            cm = metrics.confusion_matrix(y_true, y_pred)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

            if hasattr(model, 'event_decoder') and model.event_decoder.n_cifs == n_classes:

                y_event_score = (np.concatenate(y_event_score_list)[masks, :])
                y_event_score = y_event_score/y_event_score.sum(1)[:, None]
                y_event_pred = np.argmax(y_event_score, 1)

                if n_classes == 2:

                    dict_metrics.update({

                        'NextType(MC)/auc-ovo-weighted-CIF': metrics.roc_auc_score(y_true, y_score[:, 0], multi_class='ovo', average='weighted'),
                        'NextType(MC)/f1-weighted-CIF': metrics.f1_score(y_true,  y_event_pred, average='weighted', zero_division=0),
                    })
                else:
                    dict_metrics.update({

                        'NextType(MC)/auc-ovo-weighted-CIF': metrics.roc_auc_score(y_true, y_event_score, multi_class='ovo', average='weighted', labels=torch.arange(n_classes), ),
                        'NextType(MC)/f1-weighted-CIF': metrics.f1_score(y_true,  y_event_pred, average='weighted', labels=torch.arange(n_classes), zero_division=0),
                    })

            if n_classes == 2:
                dict_metrics.update({
                    # 'NextType(MC)/f1-micro': metrics.f1_score(y_true, y_pred, labels= torch.arange(n_classes) ,average='micro', zero_division=0),
                    # 'NextType(MC)/f1-macro': metrics.f1_score(y_true, y_pred,average='weighted'),
                    'NextType(MC)/f1-weighted': metrics.f1_score(y_true, y_pred, average='weighted'),
                    'NextType(MC)/precision-weighted': metrics.precision_score(y_true, y_pred),
                    'NextType(MC)/recall-weighted': metrics.recall_score(y_true, y_pred),

                    # 'NextType(MC)/auc-ovo-macro': metrics.roc_auc_score(y_true, y_score[:,0], multi_class='ovo',average='macro'),
                    'NextType(MC)/auc-weighted': metrics.roc_auc_score(y_true, y_score[:, 0], multi_class='ovo', average='weighted'),
                    # 'auc-ovo-weighted-stupid': metrics.roc_auc_score(y_true, y_pred_stupid, multi_class='ovo',average='weighted',labels= torch.arange(n_classes)),



                    'NextType(MC)/acc': metrics.accuracy_score(y_true, y_pred, normalize=True),

                    'ConfMat': cm_display,


                })
            else:
                dict_metrics.update({
                    # 'NextType(MC)/f1-micro': metrics.f1_score(y_true, y_pred, labels= torch.arange(n_classes) ,average='micro', zero_division=0),
                    'NextType(MC)/f1-macro': metrics.f1_score(y_true, y_pred, labels=torch.arange(n_classes), average='macro', zero_division=0),
                    'NextType(MC)/f1-weighted': metrics.f1_score(y_true, y_pred, labels=torch.arange(n_classes), average='weighted', zero_division=0),

                    # 'NextType(MC)/auc-ovo-macro': metrics.roc_auc_score(y_true, y_score, multi_class='ovo',average='macro',labels= torch.arange(n_classes)),
                    'NextType(MC)/auc-weighted': metrics.roc_auc_score(y_true, y_score, multi_class='ovo', average='weighted', labels=torch.arange(n_classes)),
                    # 'auc-ovo-weighted-stupid': metrics.roc_auc_score(y_true, y_pred_stupid, multi_class='ovo',average='weighted',labels= torch.arange(n_classes)),



                    'NextType(MC)/acc': metrics.accuracy_score(y_true, y_pred, normalize=True),

                    'ConfMat': cm_display,


                })

        out['y_pred'] = y_pred
        out['y_true'] = y_true
        out['y_score'] = y_score

        out['y_true_list'] = y_true_list
        out['masks'] = masks

    # label prediction
    if hasattr(model, 'pred_label'):

        # state_label_red = state_label.sum(1).bool().int()[:,None,None] # [B,1,1]
        # state_label_loss,(y_state_pred, y_state_true, y_state_score) = Utils.state_label_loss(state_label_red,model.y_label, non_pad_mask)
        # # total_loss.append(state_label_loss*opt.w_sample_label)
        # y_state_pred_list.append(y_state_pred) # [*]
        # y_state_true_list.append(y_state_true) # [*]
        # y_state_score_list.append(y_state_score) # [*] it is binary

        # y_state_pred = (np.concatenate(y_state_pred_list)) # [*]
        # y_state_true = (np.concatenate(y_state_true_list))
        # y_state_score = (np.concatenate(y_state_score_list))

        # y_state_pred = (np.concatenate(y_state_pred_list) [masks]) # [*]
        # y_state_true = (np.concatenate(y_state_true_list) [masks])
        # y_state_score = (np.concatenate(y_state_score_list) [masks])

        y_state_pred = (np.concatenate(y_state_pred_list))  # [*]
        y_state_true = (np.concatenate(y_state_true_list))
        y_state_score = (np.concatenate(y_state_score_list))

        out['y_state_pred'] = y_state_pred
        out['y_state_true'] = y_state_true
        out['y_state_score'] = y_state_score

        dict_metrics.update({
            'pred_label/AUROC': metrics.roc_auc_score(y_state_true, y_state_score),
            'pred_label/AUPRC': metrics.average_precision_score(y_state_true, y_state_score),
            'pred_label/f1-binary': metrics.f1_score(y_state_true, y_state_pred, average='binary', zero_division=0),

            'pred_label/loss': total_label_state/total_num_event,
            'pred_label/recall-binary': metrics.recall_score(y_state_true, y_state_pred, average='binary', zero_division=0),
            'pred_label/precision-binary': metrics.precision_score(y_state_true, y_state_pred, average='binary', zero_division=0),

            'pred_label/ACC': metrics.accuracy_score(y_state_true, y_state_pred),


            # 'pred_label/sum_r_enc':torch.concat(r_enc_list, dim=1).sum().item(),
            # 'pred_label/r_enc_zero': a,
        })

    return dict_metrics, out


