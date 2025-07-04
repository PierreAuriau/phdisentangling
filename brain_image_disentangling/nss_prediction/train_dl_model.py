# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from datasets import NSSDataset
from dl_model import DLModel
from loggers import setup_logging
from config import Config

config = Config()

def train_one_model(chkpt_dir, preproc, label="NSS", nb_folds=10, nb_epochs=50):
    chkpt_dir = os.path.join(config.path2models, preproc, chkpt_dir)
    os.makedirs(chkpt_dir, exist_ok=True)

    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "logs.log"))
      
    for fold in range(nb_folds):

        print(f"# Fold {fold}")
        # Model
        model = DLModel(n_embedding=256, architecture="densenet")

        # data loaders
        train_dataset = NSSDataset(split="train", fold=fold, preproc=preproc, target=label)
        train_loader = DataLoader(train_dataset, batch_size=16,
                                  shuffle=True, num_workers=1)
        
        test_dataset = NSSDataset(split="test", fold=fold, preproc=preproc, target=label)
        test_loader = DataLoader(test_dataset, batch_size=16,
                                 shuffle=False, num_workers=1)

        # train & test model
        model.fit(train_loader, chkpt_dir=chkpt_dir, label=label, preproc=preproc,
                  nb_epochs=nb_epochs, fold=fold, lr=1e-4, weight_decay=5e-5)
        model.test(test_loader, epoch=nb_epochs-1, label=label, fold=fold,
                   preproc=preproc, chkpt_dir=chkpt_dir, save_y_pred=True)
        
        """
        for epoch in range(0, 100, 99):
            model.test(test_loader, epoch=epoch, label=label, fold=fold,
                        preproc=preproc, chkpt_dir=chkpt_dir)
        """

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                     description='Train/test DL model.')
    parser.add_argument("--preproc", required=True, choices=["skeleton", "vbm"],
                        help="Preprocessing images on which the model is trained.")
    parser.add_argument("--chkpt_dir", required=True, help="Directory where data is saved.")
    parser.add_argument("--label", default="NSS", help="Label to predict. Default is NSS.")
    args = parser.parse_args(argv)    
    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    train_one_model(chkpt_dir=args.chkpt_dir, preproc=args.preproc,
                    label=args.label)