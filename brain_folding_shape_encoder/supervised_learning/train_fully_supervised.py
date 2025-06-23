# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import re
import logging

import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, \
                            mean_squared_error, r2_score, roc_auc_score

# from project
from log import setup_logging
from model import DLModel
from datamanager import DataManager
from config import Config

config = Config()

   
def train_model(chkpt_dir, dataset, smooth_preproc=False,
                architecture="densenet", n_embedding=256,
                fold=None, nb_epochs=100, lr=1e-4, weight_decay=5e-3,
                batch_size=32, num_workers=8):
    
    model = DLModel(architecture=architecture, n_embedding=n_embedding)
    
    datamanager = DataManager(dataset=dataset, label="diagnosis", fold=fold,
                              smooth_preproc=False, two_views=False, data_augmentation=None)  
    
    train_loader = datamanager.get_dataloader(split="train", shuffle=True,
                                            batch_size=batch_size,
                                            num_workers=num_workers)
    val_loader = datamanager.get_dataloader(split="validation",
                                            batch_size=batch_size,
                                            num_workers=num_workers)
    pos_weight = 1 / datamanager.dataset["train"].target.mean() - 1
    logs = {"dataset": dataset, "label": "diagnosis", 
            "smooth_preproc": smooth_preproc}
    if fold is not None:
        logs["fold"] = fold
    
    model.fit(train_loader, val_loader, pos_weight=pos_weight,
              nb_epochs=nb_epochs, chkpt_dir=chkpt_dir, logs=logs,
              lr=lr, weight_decay=weight_decay)
    
    train_loader = datamanager.get_dataloader(split="train", shuffle=False,
                                    batch_size=batch_size,
                                    num_workers=num_workers)
    internal_test_loader = datamanager.get_dataloader(split="internal_test",
                                                    batch_size=30,
                                                    num_workers=num_workers)
    external_test_loader = datamanager.get_dataloader(split="external_test",
                                                      batch_size=30,
                                                      num_workers=num_workers)
    metrics = {
        "roc_auc": lambda y_true, y_pred: roc_auc_score(y_true=y_true, y_score=y_pred),
        "balanced_accuracy": lambda y_true, y_pred: balanced_accuracy_score(y_true=y_true,
                                                                            y_pred=(y_pred > 0.5).astype(int))
    }
    model.test(loaders=[train_loader, val_loader,
                        internal_test_loader, external_test_loader],
                splits=["train", "validation", "internal_test", "external_test"],
                epoch=(nb_epochs-1), metrics=metrics, chkpt_dir=chkpt_dir,
                logs=logs)

def train(params):
    chkpt_dir = params["chkpt_dir"]
    nb_folds = params.get("nb_folds")
    n_embedding = params.get("n_embedding", config.n_embedding)

    for dataset in ("asd", "bd", "scz"):
        logger.info(f"Dataset: {dataset}")
        for fold in range(nb_folds):
            logger.info(f"Fold: {fold}")
            chkpt_dir_dt = os.path.join(chkpt_dir, dataset, f"fold-{fold}")
            os.makedirs(chkpt_dir_dt, exist_ok=True)
            
            train_model(chkpt_dir_dt, dataset,
                        n_embedding=n_embedding, 
                        fold=fold,
                        architecture=params.get("architecture", config.architecture),
                        smooth_preproc=params.get("smooth_preproc", config.smooth_preproc),
                        nb_epochs=params.get("nb_epochs", config.nb_epochs_ft), 
                        lr=params.get("lr", config.lr_ft), 
                        weight_decay=params.get("weight_decay", config.weight_decay_ft),
                        batch_size=params.get("batch_size", config.batch_size), 
                        num_workers=params.get("num_workers", config.num_workers))
            
def test(params):
    nb_folds = params.get("nb_folds")
    nb_epochs = params.get("nb_epochs", config.nb_epochs)
    smooth_preproc=params.get("smooth_preproc", config.smooth_preproc)

    for dataset in ("asd", "bd", "scz"):
        logger.info(f"Dataset: {dataset}")
        
        for fold in range(nb_folds):
            logger.info(f"Fold: {fold}")
            chkpt_dir = os.path.join(params["chkpt_dir"], dataset, f"fold-{fold}")
            hyperparameters = json.load(open(os.path.join(chkpt_dir, "hyperparaters.json"), "r"))

            model = DLModel(n_embedding=hyperparameters["n_embedding"],
                            architecture=hyperparameters["architecture"])

            datamanager = DataManager(dataset=dataset, label="diagnosis", fold=fold,
                                      smooth_preproc=smooth_preproc,
                                      two_views=False, data_augmentation=None)  
            
            train_loader = datamanager.get_dataloader(split="train", shuffle=False,
                                            batch_size=32,
                                            num_workers=8)
            val_loader = datamanager.get_dataloader(split="validation",
                                            batch_size=30,
                                            num_workers=8)
            internal_test_loader = datamanager.get_dataloader(split="internal_test",
                                                            batch_size=30,
                                                            num_workers=8)
            external_test_loader = datamanager.get_dataloader(split="external_test",
                                                            batch_size=30,
                                                            num_workers=8)
            metrics = {
                "roc_auc": lambda y_true, y_pred: roc_auc_score(y_true=y_true, y_score=y_pred),
                "balanced_accuracy": lambda y_true, y_pred: balanced_accuracy_score(y_true=y_true,
                                                                                    y_pred=(y_pred > 0.5).astype(int))
            }
            logs = {"dataset": dataset, "label": "diagnosis", "smooth_preproc": smooth_preproc}
            if fold is not None:
                logs["fold"] = fold
            for epoch in range(0, nb_epochs, 10):
                model.test(loaders=[train_loader, val_loader,
                                            internal_test_loader, external_test_loader],
                                        splits=["train", "validation", "internal_test", "external_test"],
                                        epoch=epoch, metrics=metrics, chkpt_dir=chkpt_dir,
                                        logs=logs)

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chkpt_dir", required=True, type=str,
    help="Checkpoint dir where all the logs are stored. List of existing checkpoint directories: " \
        + " - ".join(os.listdir(config.path_to_models)))
    parser.add_argument("-t", "--train", action="store_true", help="Train the model")
    parser.add_argument("-e", "--test", action="store_true", help="Test the model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbosity")
    args, unknownargs = parser.parse_known_args(argv)
    params = {}
    for i in range(0, len(unknownargs), 2):
        key = re.search("--([a-z_]+)", unknownargs[i])[1]
        params[key] = eval(unknownargs[i+1])
    return args, params


if __name__ == "__main__":

    args, params = parse_args(sys.argv[1:])
    chkpt_dir = os.path.join(config.path_to_models, args.chkpt_dir)
    params["chkpt_dir"] = chkpt_dir
    os.makedirs(chkpt_dir, exist_ok=True)
    level = "debug" if args.verbose else "info"
    setup_logging(level=level, 
                  logfile=os.path.join(chkpt_dir, "train_model.log"))
    logger = logging.getLogger("train")
    if args.train:
        train(params)
    if args.test:
        test(params)
    