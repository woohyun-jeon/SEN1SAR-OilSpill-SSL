import csv
import yaml
import random
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch


# load dataset paths
def load_dataset_ids(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]


# load configuration
def load_config(cfg_path):
    with open(cfg_path, 'r') as file:
        config = yaml.safe_load(file)

    return config


# set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# get class weights
def calculate_class_weights(dataset):
    total_pixels = 0
    oil_pixels = 0
    for _, mask, _ in dataset:
        total_pixels += mask.numel()
        oil_pixels += mask.sum().item()

    oil_ratio = oil_pixels / total_pixels

    return torch.tensor([1 / (1 - oil_ratio), 1 / oil_ratio])


def calculate_metrics(preds, targets):
    preds = (preds > 0.5).astype(float)
    preds = preds.flatten()
    targets = targets.flatten()

    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    iou = np.logical_and(preds, targets).sum() / np.logical_or(preds, targets).sum()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou
    }


def save_metrics_to_csv(metrics, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'accuracy', 'precision', 'recall', 'f1_score', 'iou'])
        writer.writeheader()
        for epoch, metric in enumerate(metrics, 1):
            row = {'epoch': epoch, **metric}
            writer.writerow(row)