import csv
import pickle
import numpy as np
from utils.helper import writeCSV
import pandas as pd
from utils.helper import *

def finalizeResult(src):
    path_default = getPath()
    print(src)
    path, filename, ext = splitFileName(src)
    read_file = pd.read_csv(f'{src}', header=None, index_col=False)
    # print(read_file)
    columns = ['date', 'model', 'dataset', 'optimizer', 'dataset_name', 'batch_size', 'index', 'train_acc', 'train_f1', 'val_acc', 'val_f1', 'loss', 'test_acc', 'test_f1', 'mac_roc_auc', 'mic_roc_auc', 'mic_prc_auc']
    columns_mean = ['train_acc', 'train_f1', 'val_acc', 'val_f1', 'loss', 'test_acc', 'test_f1', 'mac_roc_auc', 'mic_roc_auc', 'mic_prc_auc']
    columns_sub = ['date', 'model', 'dataset', 'dataset_name', 'batch_size', 'train_acc', 'train_f1', 'val_acc', 'val_f1', 'loss', 'test_acc', 'test_f1', 'mac_roc_auc', 'mic_roc_auc', 'mic_prc_auc']
    columns_order = ['date', 'model', 'dataset', 'dataset_name', 'batch_size', 'test_f1', 'loss', 'test_acc', 'mac_roc_auc', 'mic_roc_auc', 'mic_prc_auc', 'train_acc', 'train_f1', 'val_acc', 'val_f1']
    read_file.columns = columns
    df = read_file
    
    # 추후 batch 통합 코드 필요
    dst = f'{path_default}/{path}/clean/test_{filename}.xlsx'

    read_file.to_excel (dst, index = None, header=True)
    to_check = list(set(df['dataset_name']))
    df_total = pd.DataFrame([], columns=columns_sub)
    for dataset_name in to_check:
        df_sub = df[df['dataset_name']==dataset_name]
        if df_sub.shape[0] > 5:
            if 'batch_size' in df_sub.columns:
                batch_sizes = list(set(df_sub['batch_size']))
                for batch_size_ in batch_sizes:
                    df_sub_ = df_sub[df_sub['batch_size']==batch_size_]
                    date = df_sub[columns[0]].values[0]
                    model = df_sub[columns[1]].values[0]
                    dataset = df_sub[columns[2]].values[0]
                    dataset_name = df_sub[columns[4]].values[0]
                    batch_size = batch_size_
                    list_row = [date, model, dataset, dataset_name, batch_size]
                    for column in columns_mean:
                        list_row.append(f'{df_sub_[column].mean():.04f}')
                    df_row = pd.DataFrame([list_row], columns=columns_sub)
                    df_total = pd.concat([df_total,df_row], ignore_index=True)
        else :
            date = df_sub[columns[0]].values[0]
            model = df_sub[columns[1]].values[0]
            dataset = df_sub[columns[2]].values[0]
            dataset_name = df_sub[columns[4]].values[0]
            batch_size = df_sub[columns[5]].values[0]
            list_row = [date, model, dataset, dataset_name, batch_size]
            for column in columns_mean:
                list_row.append(f'{df_sub[column].mean():.04f}')
            df_row = pd.DataFrame([list_row], columns=columns_sub)
            df_total = pd.concat([df_total,df_row], ignore_index=True)

    df_total = df_total[columns_order]

    with pd.ExcelWriter(dst, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
        df_total.to_excel(writer, sheet_name='total', index=False)

import glob
from utils.helper import *
dir = './result'
files = glob.glob(f'{dir}/*.csv')

path_default = getPath()
path, filename, ext = splitFileName(files[0])


cleanDirs([f'{path_default}/{path}/clean'])
for file in files:
    finalizeResult(file)