import torch
import pandas as pd
from sklearn import metrics
import numpy as np
import json


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def multilabel_acc(pred_label, true_label):
    return metrics.accuracy_score(true_label, pred_label)


def multilabel_metric(pred_label, true_label):
    mm = {
        'acc': metrics.accuracy_score(true_label, pred_label),
        'precision': metrics.precision_score(true_label, pred_label,  average='micro'),
        'reacll': metrics.recall_score(true_label, pred_label, average='micro'),
        'f1': metrics.f1_score(np.array(true_label), np.array(pred_label), average='micro'),
        'confusion_matrix': metrics.confusion_matrix(true_label, pred_label).tolist()
    }
    mcm = metrics.multilabel_confusion_matrix(true_label, pred_label)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    mm['acc_list'] = list((tp + tn) / (tp + tn + fp + fn))
    mm['recall_list'] = list(tp / (tp + fn))
    mm['precision_list'] = list(tp / (tp + fp))
    return mm


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


def save_csv(cache, csv_path):
    colums = list(cache.keys())
    values = list(cache.values())
    values_T = list(map(list, zip(*values)))
    save = pd.DataFrame(columns=colums, data=values_T)
    f1 = open(csv_path, mode='w', newline='')
    save.to_csv(f1, encoding='gbk', index=False)
    f1.close()


def read_csv(csv_path):
    pd_data = pd.read_csv(csv_path, sep=',', header='infer', usecols=['Value'])
    # pd_data['Status'] = pd_data['Status'].values
    return pd_data


def save_json(cache, json_path):
    # 保存文件
    tf = open(json_path, "w")
    tf.write(json.dumps(cache))
    tf.close()


def read_json(json_path):
    # 读取文件
    tf = open(json_path, "r")
    new_dict = json.load(tf)
    return new_dict