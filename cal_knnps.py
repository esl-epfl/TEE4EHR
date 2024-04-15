

# general
import wandb
from MulticoreTSNE import MulticoreTSNE as TSNE
from tsnecuda import TSNE
import re
from sklearn import metrics
from tqdm import tqdm
import Utils
import transformer.Constants as Constants
import torch.optim as optim
import torch.nn as nn
import torch
import time
import argparse
import numpy as np
import pandas as pd
import math
# import matplotlib.pyplot as plt
import os
import shutil
import pickle

import Main

# plotly
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

# import custom libraries
import sys
# sys.path.append("C:\\DATA\\Tasks\\lib\\hk")
# import hk_utils

# folder paths
ADD_DATA = "C:\\DATA\\data\\raw\\mimic4\\lookup\\"
ADD_DATA_proc = "C:/DATA/data/processed/"


PATH_PAPER = "C:\\DATA\\Tasks\\220704\\Alternate-Transactions-Articles-LaTeX-template\\images\\"


PATH_SYS = "/mlodata1/hokarami/tedam/"


# libraries for THP

# from torch.utils.tensorboard import SummaryWriter


# from preprocess.Dataset import get_dataloader, get_dataloader2
# from transformer.Models import Transformer
# from transformer.hk_transformer import Transformer

# from torchinfo import summary

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# torch.cuda.empty_cache()
# torch.cuda.memory_allocated()
# torch.cuda.memory_reserved()

# from hk_pytorch import save_checkpoint,load_checkpoint
# import hk_pytorch


# from custom2 import myparser


# wandb.login()
api = wandb.Api()

os.environ["WANDB_API_KEY"] = "0f780ac8a470afe6cb7fc474ff3794772c660465"


def build_df(out, opt, X_tsne=None, TSNE_LIMIT=5000):
    if 'y_state_true' in out:

        y_state_pred = out['y_state_pred']
        y_state_true = out['y_state_true']
        y_state_score = out['y_state_score']
    else:
        y_state_pred = None
        y_state_true = None
        y_state_score = None

    y_pred = out['y_pred']
    y_true = out['y_true']
    y_score = out['y_score']

    df = pd.DataFrame()

    if X_tsne is not None:
        df['x'] = X_tsne[:, 0]
        df['y'] = X_tsne[:, 1]
    else:
        df['x'] = y_state_true
        TSNE_LIMIT = 100000000

    # df['x']=X_te_tsne[:,0]
    # df['y']=X_te_tsne[:,1]

    # df['x']=X_dam_tsne[:,0]
    # df['y']=X_dam_tsne[:,1]

    df['color'] = 0
    df['id'] = np.arange(len(df))

    if y_state_true is not None:

        TP = (y_state_true[:TSNE_LIMIT]*y_state_pred[:TSNE_LIMIT]) == 1
        FN = (y_state_true[:TSNE_LIMIT]-y_state_pred[:TSNE_LIMIT]) == 1
        FP = (y_state_true[:TSNE_LIMIT]-y_state_pred[:TSNE_LIMIT]) == -1

        TN = (y_state_true[:TSNE_LIMIT]+y_state_pred[:TSNE_LIMIT]) == 0

        FP_FN = (y_state_true[:TSNE_LIMIT]+y_state_pred[:TSNE_LIMIT]) == 1
        TP_TN = (y_state_true[:TSNE_LIMIT]-y_state_pred[:TSNE_LIMIT]) == 0

        # df.loc[TN, 'color']='True Negatives'
        df.loc[TP, 'color'] = 'True Positives'
        df.loc[FN, 'color'] = 'False Negatives'
        df.loc[FP, 'color'] = 'False Positives'
        df.loc[TN, 'color'] = 'True Negatives'

        df.loc[TP_TN, 'color_true_pred'] = 'True Predicted'
        df.loc[FP_FN, 'color_true_pred'] = 'False Predicted'

        df.loc[y_state_true[:TSNE_LIMIT].astype(
            bool).flatten(), 'color_true'] = 'Positive Samples'
        df.loc[~y_state_true[:TSNE_LIMIT].astype(
            bool).flatten(), 'color_true'] = 'Negative Samples'

        df.loc[y_state_pred[:TSNE_LIMIT].astype(
            bool).flatten(), 'color_pred'] = 'Positive Predicted'
        df.loc[~y_state_pred[:TSNE_LIMIT].astype(
            bool).flatten(), 'color_pred'] = 'Negative Predicted'

        # df.loc[y_state_pred[:TSNE_LIMIT].astype(bool).flatten(), 'color_true_pred']='Positive Predicted'
        # df.loc[~y_state_pred[:TSNE_LIMIT].astype(bool).flatten(), 'color_true_pred']='Negative Predicted'

    df['i_b'] = df['id'].apply(lambda x: int(x / opt.batch_size))
    df['i'] = df['id'].apply(lambda x: x % opt.batch_size)

    t_max = np.concatenate([t.max(1)[0] for t in out['event_time_list']])

    df['color_t_max'] = t_max
    if len(out['list_log_sum']) > 0:
        df['color_log_sum'] = np.concatenate(
            out['list_log_sum'], axis=0) / t_max
        df['color_integral_'] = np.concatenate(
            out['list_integral_'], axis=0) / t_max
    return df


def event2mat(event_type, event_time, P):

    m = event_type.sum(1) > 0  # False are masked
    e = event_type[m, :]
    t = event_time[m]
    # print(t)

    indices = e.nonzero()
    indices[:, 0] = t[indices[:, 0]].int()
    # print(t[indices[:,0]])
    # print(indices[:,0])

    M = torch.zeros((P, e.shape[-1]))
    M[indices[:, 0], indices[:, 1]] = 1

    return M.detach().cpu().numpy().transpose()  # [M,P]


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


def read_from_wandb(run_path, consider_sample_labels=False):

    run = api.run(run_path)

    # for file in run.files():
    #     file.download(replace=True,root=f'./local/{run_path}/')
    for file in run.files():
        if 'best_model' in file.name or 'opt.pkl' in file.name:

            try:
                file.download(replace=True, root=f'.local/{run_path}')

            except:
                print('ERROR')
                pass
    opt = pickle.load(open(f'.local/{run_path}/opt.pkl', 'rb'))

    if consider_sample_labels:
        opt.sample_label = True

    opt = Main.config(opt, justLoad=True)
    if not hasattr(opt, 'diag_offset'):
        opt.diag_offset = 1
        print('ATTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT')
    checkpoint = torch.load(f'.local/{run_path}/best_model.pkl')

    model = Main.ATHP(
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

    _ = model.to(opt.device)
    _ = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    _ = model.eval()

    return model, opt, run


def find_knn_pids(X, id_origin, n_knn=10):

    # X [N,d] np.array
    r = np.sqrt(((X-X[id_origin])**2).sum(-1))
    # knn_pids = list(np.argpartition(r,n_knn)[:n_knn])
    knn_pids = np.argsort(r)[:n_knn]
    if id_origin not in knn_pids:
        print('bad ', id_origin, knn_pids)

    else:
        # knn_pids.remove(id_origin)
        aa = 1

    # print(X.shape)
    # term

    # x0 = df.iloc[id_origin]['x']
    # y0 = df.iloc[id_origin]['y']

    # df['r'] = df.apply(lambda row: ( (row['x']-x0)**2 + (row['y']-y0)**2  ), axis=1)

    # knn_pids = list(df.nsmallest(n_knn,'r').index)

    return knn_pids


def cal_similarity(df, out, pid_origin, knn_pids):

    list_summary = []
    list_t_max = []
    for pid in knn_pids:

        i_b = df.iloc[pid]['i_b']
        i = df.iloc[pid]['i']

        ev = out['event_type_list'][i_b][i]
        t = out['event_time_list'][i_b][i]
        st = out['state_time_list'][i_b][i]

        P = st.int().max().item() + 1

        M = event2mat(ev, t, P)

        # if ev.sum()==0:
        #     print(pid)
        #     continue
        #     # print(t.shape,ev.sum())
        vector = M.sum(1)/M.shape[1]*24

        # if (vector.sum()==0):
        #     print(M.shape,t.max(),ev.sum(),vector.sum(),'binjour',st.max())
        #     # print(pid_origin,pid)
        #     # print('shit')
        #     if pid==pid_origin:
        #         print('WTF ',pid)
        #         raise 'pid_origin has not pattern!'
        #     # bad_ids.append(pid)
        #     continue

        list_summary.append(vector)
        list_t_max.append(t.max().item())

    dotp = [np.dot(j, list_summary[0])/(np.linalg.norm(j) *
                                        np.linalg.norm(list_summary[0])) for j in list_summary]
    sim_score = np.mean([x for x in dotp if ~np.isnan(x)]
                        )  # remove nan elements

    if np.isnan(sim_score.sum()):
        print('he', sim_score.sum())

    # print('ff',list_t_max)
    list_t_max = [x for x in list_t_max if x > 0]
    temp = np.array(list_t_max)
    t_max_mae = np.mean(np.abs(temp-temp[0]))
    # t_max_score = [x-list_t_max[0] for x in list_t_max[1:]]

    return sim_score, list_summary, t_max_mae


def remove_ids_with_no_pattern(df, out):

    bad_pids = []
    for pid in df.id:

        i_b = df.iloc[pid]['i_b']
        i = df.iloc[pid]['i']

        # ev = out['event_type_list'][i_b][i]
        t = out['event_time_list'][i_b][i]
        # st=out['state_time_list'][i_b][i]

        if t.max() == 0:
            bad_pids.append(pid)

    print(f"{len(bad_pids)} patients were removed due to no existing pattern")
    df_out = df.loc[~df.id.isin(bad_pids)]
    df_out.loc[:, 'id'] = np.arange(len(df_out))

    return df_out, bad_pids


if __name__ == '__main__':
    print('hi')
    api = wandb.Api()
    runs = api.runs("hokarami/TEEDAM_supervised")
    df_filt = dl_runs(runs, selected_tag='RD74-TableIII-v5')
    len(df_filt)

    # obtaining runs
    df_config = pd.DataFrame([{k: v for k, v in x.items()}
                             for x in df_filt.config])
    df_summary = pd.DataFrame([{k: v for k, v in x.items()}
                              for x in df_filt.summary])
    df_path = df_filt.path.apply(lambda x: '/'.join(x))
    df_con = pd.concat([df_config, df_summary, df_path], axis=1)
    len(df_con)

    df_con['transfer_learning'].unique()

    if 'knn-ps-mean' in df_con:
        q = ((df_con['transfer_learning'] == 'DO') & (df_con['knn-ps-mean'].isnull())
             ) | ((df_con['INPUT'] == 'DAM') & (df_con['knn-ps-mean'].isnull()))
    else:
        q = df_con['transfer_learning'].astype(bool)+True
    q = df_con['transfer_learning'].astype(bool)+True

    df_con = df_con[q].iloc[:]
    len(df_con)
    run_paths = df_con.path.tolist()

    # computing

    # run_path = "hokarami/TEEDAM_supervised/3drow7po" # https://wandb.ai/hokarami/TEEDAM_supervised/runs/3drow7po/overview?workspace=user-g-hojatkarami

# run_paths = ["hokarami/TEEDAM_supervised/3drow7po"]
n_knn = 10
temp = list()
bad_runs = list()
bad_cal_sim = 0
for run_path in tqdm(run_paths[:], leave=False):
    api = wandb.Api()
    run = api.run(run_path)
    try:
        model1, opt1, run1 = read_from_wandb(
            run_path, consider_sample_labels=True)

        dict_metrics1, out1 = Main.valid_epoch_tsne(
            model1, opt1.validloader, opt1.pred_loss_func, opt1)
        # X_tsne1, X_tsne_split1 = compute_tsne(out1['r_enc_list'], model1)
        res_labels = opt1.dict_map_events.keys()
        df1 = build_df(out1, opt1, X_tsne=None)

        df1, bad_pids = remove_ids_with_no_pattern(df1, out1)

        pid_all = list(df1.id)

        pid_positive = list(df1[df1.color_true == 'Positive Samples'].id)

        list_sim_score1 = []
        list_t_max_rmse1 = []

        # X1 = np.concatenate(out1['r_enc_list'],axis=0)[:,:]

        # only DAM embeddings
        X1 = np.concatenate(out1['r_enc_list'], axis=0)[
            :, model1.d_out_te:model1.d_out_te+model1.d_out_dam]

        # only TEE embeddings
        X1 = np.concatenate(out1['r_enc_list'], axis=0)[:, :model1.d_out_te]

        # TEE+DAM embeddings
        X1 = np.concatenate(out1['r_enc_list'], axis=0)[
            :, :model1.d_out_te+model1.d_out_dam]
        X1 = np.delete(X1, bad_pids, axis=0)

        for pid in tqdm(pid_positive):
            id_origin = pid

            knn_pids = find_knn_pids(X1, id_origin, n_knn=n_knn)
            try:
                sim_score1, list_summary, t_max_mae1 = cal_similarity(
                    df1, out1, pid, knn_pids)

                if (not np.isnan(sim_score1)):
                    list_sim_score1.append(sim_score1)
                    list_t_max_rmse1.append(t_max_mae1)
            except:
                bad_cal_sim += 1

        # np.mean(list_sim_score1), np.std(list_sim_score1)

        run.summary['knn-ps-version'] = 1
        run.summary['bad_pids'] = len(bad_pids)
        run.summary['knn-ps-mean'] = np.mean(list_sim_score1)
        run.summary['knn-ps-std'] = np.std(list_sim_score1)
        run.summary['t_max_mae-mean'] = np.mean(list_t_max_rmse1)
        run.summary['t_max_mae-std'] = np.std(list_t_max_rmse1)
        run.summary.update()

        temptemp = dict()
        temptemp['path'] = run_path
        temptemp['knn-ps-mean'] = np.mean(list_sim_score1)
        temptemp['knn-ps-std'] = np.std(list_sim_score1)
        temp.append(temptemp)
    except:
        bad_runs.append(run_path)
        run.summary['knn-ps-mean'] = -0.01
        continue
