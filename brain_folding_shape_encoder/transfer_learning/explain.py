# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import argparse
import itertools
import time
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, \
                            mean_squared_error, r2_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# project imports
from dataset import ClinicalDataset
from data_augmentation import ToTensor, Cutout

# from project
from log import setup_logging
from model import BTModel
from config import Config

config = Config()

def patch_occlusion(chkpt_dir, dataset, epoch=-1, fold=None):
        
    metrics = {
        "roc_auc": lambda y_true, y_pred: roc_auc_score(y_true=y_true, y_score=y_pred),
        "balanced_accuracy": lambda y_true, y_pred: balanced_accuracy_score(y_true=y_true,
                                                                            y_pred=(y_pred > 0.5).astype(int))
    }
    patch_size = 16
    step_size = 4
    img_size = (128, 160, 128)
    label = "diagnosis"
    splits = ["internal_test", "external_test"]
    target_mapping = {"control": 0, 
                    "asd": 1,
                    "bd": 1, "bipolar disorder": 1, "psychotic bd": 1, 
                    "scz": 1}
    print(f"DATASET: {dataset} / fold: {fold}")
    chkpt_dir_dt = os.path.join(chkpt_dir, dataset, f"fold-{fold}")
    with open(os.path.join(chkpt_dir_dt, "hyperparameters.json"), "r") as json_file:
        hyperparameters = json.load(json_file)
    n_embedding = hyperparameters.get("n_embedding", config.n_embedding)
    model = BTModel(n_embedding=n_embedding, classifier=True)
    if epoch < 0:
        epoch = hyperparameters.get("nb_epochs", config.nb_epochs_ft) + epoch
        
    df_xai = None
    tic = time.time()
    for x, y, z in itertools.product(range(patch_size//2, img_size[0]-patch_size//2+1, step_size), 
                                range(patch_size//2, img_size[1]-patch_size//2+1, step_size), 
                                range(patch_size//2, img_size[2]-patch_size//2+1, step_size)):
        print(f"Patch({x}, {y}, {z})")
        tr = transforms.Compose([Cutout(patch_size=patch_size, localization=(x, y, z), random_size=False), ToTensor()])

        datasets = [ClinicalDataset(split=s, label=label, fold=fold,
                                    dataset=dataset, transforms=tr,
                                    target_mapping=target_mapping)
                    for s in splits]
        
        loaders = [DataLoader(d, drop_last=False, batch_size=30, 
                            num_workers=4)
                    for d in datasets]

    
        logs = model.test_classifier(loaders=loaders, splits=splits, 
                              epoch=epoch, metrics=metrics, chkpt_dir=chkpt_dir_dt,
                              logs={"dataset": dataset, "label": "diagnosis", 
                                    "x_occluded": x, "y_occluded": y, "z_occluded": z},
                              return_logs=True)
        logs_df = pd.DataFrame(logs)
        if df_xai is None:
            df_xai = logs_df
            df_xai.to_csv(os.path.join(chkpt_dir_dt, f"classifier_fold-{fold}_epoch-{epoch}_xai.csv"), sep=",", index=False)
        else:
            df_xai = pd.concat([df_xai, logs_df], axis=0, ignore_index=True, sort=False)
            if time.time() - tic > 3600: # saving every hour
                df_xai.to_csv(os.path.join(chkpt_dir_dt, f"classifier_fold-{fold}_epoch-{epoch}_xai.csv"), sep=",", index=False)
                tic = time.time()
    # final save
    df_xai.to_csv(os.path.join(chkpt_dir_dt, f"classifier_fold-{fold}_epoch-{epoch}_xai.csv"), sep=",", index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chkpt_dir", type=str, required=True,
                        help="Checkpoint directory")    
    parser.add_argument("-f", "--fold", type=int, default=None,
                        help="fold for the explanation")
    parser.add_argument("-d", "--dataset", type="str", required=True, 
                        choices=["asd", "bd", "scz"])
    parser.add_argument("-e", "--epoch", type=int, default=-1,
                        help="Epoch to be explained")
    args = parser.parse_args()

    setup_logging(level="info",
                  logfile=os.path.join(args.chkpt_dir,  "explain.log"))

    patch_occlusion(chkpt_dir=args.chkpt_dir,
                    fold=args.fold,
                    dataset=args.dataset,
                    epoch=args.epoch)