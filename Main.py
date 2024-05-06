import datetime
import os
import shutil
import sys

from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn import metrics
# from torch.utils.tensorboard import SummaryWriter
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

from preprocess.Dataset import get_dataloader

from transformer.Models import TEEDAM, align
from tqdm import tqdm


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
        # # opt.writer.add_figure('matplotlib', fig, i_epoch)
        dict_metrics.pop('ConfMat')
        # plt.close()

    if 'time_gap_data' in dict_metrics:
        fig, ax = plt.subplots(figsize=(10, 10))
        _ = ax.scatter(dict_metrics['time_gap_data'][0]
                       [1:], dict_metrics['time_gap_data'][1][:-1])
        _ = ax.plot([0, 2], [0, 2], 'r-')
        # # opt.writer.add_figure('time_gap', fig, i_epoch)
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
            # # opt.writer.add_figure('tsne', fig, i_epoch)
        dict_metrics.pop('tsne')
    if 'pred_label/PR_curve' in dict_metrics:
        # fig = plt.figure(figsize=(8,8))
        # fig, ax = plt.subplots(figsize=(10, 10))

        wandb.log(
            {'Precision-Recall Curve': wandb.Image(dict_metrics['pred_label/PR_curve'])}, step=i_epoch)
        plt.close()
        # dict_metrics['pred_label/PR_curve'].plot(ax=ax)
        # dict_metrics['pred_label/PR_curve'].plot()
        # # opt.writer.add_figure('matplotlib', fig, i_epoch)
        dict_metrics.pop('pred_label/PR_curve')
        # plt.close()
    for k, v in dict_metrics.items():

        if isinstance(v, np.ndarray):
            # # opt.writer.add_histogram(prefix+k, v, i_epoch)
            a = 1
        else:
            # # opt.writer.add_scalar(prefix+k, v, i_epoch)
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
            if 'demo' in train_state.keys():
                train_state['demo'] = train_state['demo'][:int(
                    opt.per/100*len(train_state['demo']))]

    state_args = {'have_label': opt.sample_label, 'have_demo': opt.demo}

    trainloader = get_dataloader(train_data, data_state=train_state, bs=opt.batch_size, shuffle=True,
                                  data_label=opt.data_label, balanced=opt.balanced_batch, state_args=state_args)
    testloader = get_dataloader(test_data, data_state=test_state, bs=opt.batch_size,
                                 shuffle=False, data_label=opt.data_label, balanced=False, state_args=state_args)
    validloader = get_dataloader(valid_data, data_state=valid_state, bs=opt.batch_size,
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



    log_loss = {'loss/event_decoder': 0, 'loss/pred_next_time': 0,
                'loss/pred_next_type': 0, 'loss/pred_label': 0}
    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):



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


        """ update parameters """

        optimizer.step()



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
                total_time_sse += sse.item()  # cumulative time prediction squared-error
                total_time_sae += sae.item()  # cumulative time prediction squared-error
                # cumulative time prediction squared-error
                total_time_sse_norm += sse_norm.item()

            # label prediction
            if hasattr(model, 'pred_label') and (state_label is not None):

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
                   
                    'NextType(MC)/f1-weighted': metrics.f1_score(y_true, y_pred, average='weighted'),
                    'NextType(MC)/precision-weighted': metrics.precision_score(y_true, y_pred),
                    'NextType(MC)/recall-weighted': metrics.recall_score(y_true, y_pred),

                    'NextType(MC)/auc-weighted': metrics.roc_auc_score(y_true, y_score[:, 0], multi_class='ovo', average='weighted'),



                    'NextType(MC)/acc': metrics.accuracy_score(y_true, y_pred, normalize=True),

                    'ConfMat': cm_display,


                })
            else:
                dict_metrics.update({
                    'NextType(MC)/f1-macro': metrics.f1_score(y_true, y_pred, labels=torch.arange(n_classes), average='macro', zero_division=0),
                    'NextType(MC)/f1-weighted': metrics.f1_score(y_true, y_pred, labels=torch.arange(n_classes), average='weighted', zero_division=0),

                    'NextType(MC)/auc-weighted': metrics.roc_auc_score(y_true, y_score, multi_class='ovo', average='weighted', labels=torch.arange(n_classes)),



                    'NextType(MC)/acc': metrics.accuracy_score(y_true, y_pred, normalize=True),

                    'ConfMat': cm_display,


                })

    # label prediction
    if hasattr(model, 'pred_label'):


        y_state_pred = (np.concatenate(y_state_pred_list))  # [*]
        y_state_true = (np.concatenate(y_state_true_list))
        y_state_score = (np.concatenate(y_state_score_list))

        pr, re, _ = metrics.precision_recall_curve(y_state_true, y_state_score)
        plt.figure()
        plt.plot(re, pr)
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        dict_metrics.update({
            'pred_label/AUROC': metrics.roc_auc_score(y_state_true, y_state_score),
            'pred_label/AUPRC': metrics.average_precision_score(y_state_true, y_state_score),
            'pred_label/f1-binary': metrics.f1_score(y_state_true, y_state_pred, average='binary', zero_division=0),

            'pred_label/loss': total_label_state/total_num_event,
            'pred_label/recall-binary': metrics.recall_score(y_state_true, y_state_pred, average='binary', zero_division=0),
            'pred_label/precision-binary': metrics.precision_score(y_state_true, y_state_pred, average='binary', zero_division=0),

            'pred_label/ACC': metrics.accuracy_score(y_state_true, y_state_pred),

            'pred_label/PR_curve': plt,

            
        })

    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse, dict_metrics


def train(model, trainloader, validloader, testloader, optimizer, scheduler, pred_loss_func, opt, trial=None):


    # evaluation before training
    start = time.time()
    valid_event, valid_type, valid_time, dict_metrics_test = valid_epoch(
        model, testloader, pred_loss_func, opt)
    print('  - (PRE Testing)     loglikelihood: {ll: 8.5f}, '
          'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
          'elapse: {elapse:3.3f} min'
          .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

    # logging
    write_to_summary(dict_metrics_test, opt, i_epoch=0, prefix='Test-')


    best_metric = -100 # for early stopping
    # Initialize the early stopping counter
    early_stopping_counter = 0

    # Set the maximum number of epochs without improvement
    max_epochs_without_improvement = opt.ES_pat

    dict_time = {}

    inter_Obj_val = 0
    best_test_metric = {}
    best_valid_metric = {}

    # choosing the best metric for early stopping
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


    for epoch_i in tqdm(range(1, 1 + opt.epoch), leave=False):

        opt.i_epoch = epoch_i

        # ********************************************* Train Epoch *********************************************
        start = time.time()
        train_event, train_type, train_time, dict_metrics_train = train_epoch(
            model, trainloader, optimizer, pred_loss_func, opt)
        scheduler.step()

        # logging learning rate
        wandb.log({'LR': optimizer.param_groups[0]['lr']}, step=epoch_i)
        dict_time.update({'Time/train_epoch': ((time.time() - start) / 60)})
        

        if opt.i_epoch % opt.log_freq == 0:

            # Evaluating train dataset (uncomment if needed)
            # train_event, train_type, train_time, dict_metrics_train2 = valid_epoch(
            #     model, trainloader, pred_loss_func, opt)
            # dict_metrics_train.update(dict_metrics_train2)
            # write_to_summary(dict_metrics_train, opt,
            #                  i_epoch=opt.i_epoch, prefix='Train-')

            # Evaluating valid dataset
            start = time.time()
            valid_event, valid_type, valid_time, dict_metrics_valid = valid_epoch(
                model, validloader, pred_loss_func, opt)

            dict_time.update(
                {'Time/valid_epoch': ((time.time() - start) / 60)})
            write_to_summary(dict_metrics_valid, opt,
                             i_epoch=opt.i_epoch, prefix='Valid-')

            # Evaluating test dataset
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


                        dict_metrics_valid

                        wandb.log({('Best-Test-'+k1): v1 for k1,
                                  v1 in dict_metrics_test.items()}, step=opt.i_epoch)

                        wandb.log({('Best-Valid-'+k1): v1 for k1,
                                  v1 in dict_metrics_valid.items()}, step=opt.i_epoch)


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
                        default='TEE4EHR', help='project name in your wandb dashboard')
    parser.add_argument('-wandb_tag', type=str,
                        default='None', help='tag for wandb run')
    parser.add_argument('-user_prefix', type=str, default='DEBUG', help='add a prefix to wandb run name')


    parser.add_argument('-log_freq', type=int, default=1,
                        help='each how many epochs to log?')



    parser.add_argument('-per', type=int, default=100,
                        help='percentage of dataset to be used for training. Only for debugging')
    parser.add_argument('-unbalanced_batch', action='store_false',
                        dest='balanced_batch', help='by default balanced batch is used with respect to the label')

    parser.add_argument('-transfer_learning', action='store_true',
                        help='using transfer learning?')

    parser.add_argument('-tl_tag', default="None",
                        help='specify a wandb tag to be loaded for transfer learning')
    parser.add_argument(
        '-freeze', nargs='+', choices=['TE', 'DAM', ''], default='', help='which modules should be freezed?')

    parser.add_argument('-ES_pat', type=int, default=10,
                        help='number of epochs for early stopping')

    # data handling

    parser.add_argument('-setting', type=str, choices=['', 'raindrop'], default='', help="'' for normal, 'raindrop' for raindrop splits")
    parser.add_argument('-test_center', type=str, default='',
                        help='for EHRs, specify the test center otherwise leave empty')
    parser.add_argument('-split', type=str, default='',
                        help='split number')

    

    # General Config
    parser.add_argument('-epoch', type=int, default=40)
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument(
        '-lr_scheduler', choices=['StepLR', 'CosineAnnealingLR'], default='CosineAnnealingLR', help="learning rate scheduler")
    # parser.add_argument('-smooth', type=float, default=0.0)
    parser.add_argument('-weight_decay', type=float, default=1e-0, help='weight decay for optimizer')

    # ************************************** INPUT **************************************
    # Transformer Encoder Architecture
    parser.add_argument('-diag_offset', type=int,
                        default=1, help='MASK PARAMETER in TEE. [-2,-1,0,1] corresponds to w in [3,2,1,0] respectively. default is 1')

    parser.add_argument('-event_enc', type=int,
                        choices=[0, 1], default=1, help='consider event encoding?')

    parser.add_argument('-time_enc', type=str, choices=[
                        'sum', 'concat', 'none'], default='concat', help='strategy for time encoding')
    # TEE config
    parser.add_argument('--te_d_mark', type=int, default=8, help="encoding dimension of TEE")
    parser.add_argument('--te_d_time', type=int, default=8, help="time encoding dimension of TEE. should be the same as te_d_mark if strategy is sum")

    parser.add_argument('--te_d_rnn', type=int, default=256, help="ignore")
    parser.add_argument('--te_d_inner', type=int, default=16)
    parser.add_argument('--te_d_k', type=int, default=8, help="keys dimension of TEE")
    parser.add_argument('--te_d_v', type=int, default=8, help="values dimension of TEE")
    parser.add_argument('--te_n_head', type=int, default=4, help="number of heads in TEE")
    parser.add_argument('--te_n_layers', type=int, default=4, help="number of layers in TEE")
    parser.add_argument('--te_dropout', type=float, default=0.1, help="dropout in TEE")

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


    # State Encoder Architecture
    parser.add_argument('-state', action='store_true',
                        dest='state', help='consider DAM?')
    parser.add_argument('-demo', action='store_true',
                        dest='demo', help='consider demo variables in the dataset?')

    parser.add_argument('-num_states', type=int, default=1,help='number of state variables. will be read from the dataset.')

    # noise
    parser.add_argument('-noise', action='store_true',
                        dest='noise', help='consider noise? ignore')

    # Outputs

    # CIFs
    parser.add_argument(
        '-mod', type=str, choices=['single', 'mc', 'ml', 'none'], default='single', help='LOSS TYPE for CIFs')
    parser.add_argument('-int_dec', type=str, choices=[
                        'thp', 'sahp'], default='sahp', help='specify the inteesity decoder')
    parser.add_argument('-w_event', type=float, default=1, help="weight for event decoder loss")

    # marks
    parser.add_argument('-next_mark',  type=int,
                        choices=[0, 1], default=1, help='0: mark not detached, 1: mark detached')
    parser.add_argument('-w_class', action='store_true',
                        dest='w_class', help='consider w_class?')
    parser.add_argument('-w_pos', action='store_true',
                        dest='w_pos', help='consider w_pos?')

    parser.add_argument('-mark_detach',  type=int,
                        choices=[0, 1], default=0, help='0: mark not detached, 1: mark detached')

    # times
    parser.add_argument('-w_time', type=float, default=1.0, help="weight for time prediction loss")

    # final sample label
    
    parser.add_argument('-sample_label',  type=int,
                        choices=[0, 1, 2], default=0, help='consider labels for supervised learning task? 0: No, 1: Yes, 2) Yes, but detach the prediction head (do not update DAM, TEE)')
    parser.add_argument('-w_pos_label', type=float, default=1.0, help="weight for positive labels in BCE loss.")
    parser.add_argument('-w_sample_label', type=float, default=10000.0)

    opt = parser.parse_args()

    temp = vars(opt)

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

    # if (opt.transfer_learning != '')*0:

    #     print('### ---------TRANSFER LEARNING----------->    '+opt.transfer_learning)

    #     # load opt file
    #     with open(opt.data + opt.transfer_learning+'/opt.pkl', 'rb') as f:
    #         opt_tl = pickle.load(f)

    #     print(f"""Available modules:

    #     opt.event_enc:              {opt_tl.event_enc}
    #     opt.state:                  {opt_tl.state}

    #     opt.next_mark:              {opt_tl.next_mark}
    #     opt.sample_label:           {opt_tl.sample_label}
    #     opt.mod:                    {opt_tl.mod}
    #     opt.int_dec:                {opt_tl.int_dec}

        
        
    #     """)
    #     print(opt_tl)

    #     opt.all_transfered_modules = []

    #     if opt_tl.state:
    #         opt.all_transfered_modules.append('DAM')
    #     if opt_tl.event_enc:
    #         opt.all_transfered_modules.append('TE')
    #     # if opt.next_mark:
    #     #     opt.all_transfered_modules.append('pred_next_type')

    # else:
    #     opt_tl = opt

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
        opt.pos_weight = opt.pos_weight
        print('[Info] pos weigths:\n', opt.pos_weight)

    else:
        opt.pos_weight = torch.ones(
            opt.num_marks, device=opt.device)

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


def main():
    """ Main function. """

    print(sys.argv)

    opt = options()  # if run from command line it process sys.argv

    opt = config(opt) # some 

    torch.manual_seed(42)

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

    opt.label_loss_fun = nn.BCEWithLogitsLoss(
        reduction='none', pos_weight=torch.tensor(opt.w_pos_label, device=opt.device))

    """ prepare model """
    model = TEEDAM(
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

    if opt.transfer_learning:

        # transfer learning
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


        opt.all_transfered_modules = opt.freeze
        load_module(model, checkpoint,
                    modules=opt.all_transfered_modules, to_freeze=True)
        print('### [info] all transfered modules: ',
              opt.all_transfered_modules)


    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=opt.weight_decay)
   
    if opt.lr_scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
    elif opt.lr_scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=0.00001)

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    wandb.log({'num_params': num_params})
    
    """ train the model """    
    train(model, opt.trainloader, opt.validloader,
                            opt.testloader, optimizer, scheduler, opt.pred_loss_func, opt, None)
    
    
    shutil.copy(opt.run_path+'opt.pkl', wandb.run.dir+'/opt.pkl')
    shutil.copy(opt.run_path+'best_model.pkl', wandb.run.dir+'/best_model.pkl')


    if opt.wandb:

        # report the final validation accuracy to wandb
        wandb.run.summary["status"] = "completed"

        wandb.save('opt.pkl')
        wandb.save('best_model.pkl')
        wandb.finish(quiet=True)



if __name__ == '__main__':
    
    # setup wandb
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_KEY"))


    main()


