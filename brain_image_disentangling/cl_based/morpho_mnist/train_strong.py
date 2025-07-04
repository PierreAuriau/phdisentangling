# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train strong encoder and fine-tune weak encoder.
Weak encoder is previously trained and loaded.
"""

# import functions
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader# ignore warning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import random
# project import
from dataset import MorphoMNISTDataset
from strong_encoder import StrongEncoder
from loggers import setup_logging

from config import Config

config = Config()


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description='Train strong encoder')
    parser.add_argument(
        "-c", "--checkpoint_dir", type=str, required=True,
        help="Directory where models are saved")
    parser.add_argument(
        "-w", "--weak_size", type=int, default=32,
        help="Latent space size of the weak encoder. Default is 32.")
    parser.add_argument(
        "-d", "--pretrained_path", type=str, default=None,
        help="path toward pretrained weak encoder")
    parser.add_argument(
        "-s", "--strong_size", type=int, default=32,
        help="Latent space size of the strong encoder. Default is 32.")
    parser.add_argument(
        "-n", "--n_epochs", type=int, default=250,
        help="Number of epochs for training. Default is 250.")
    parser.add_argument(
        "-p", "--ponderation", type=float, default=100.0,
        help="Ponderation of the jem loss.")
    args = parser.parse_args(argv)

    # if not os.path.exists(args.checkpoint_dir):
    #    raise NotADirectoryError("Checkpoint directory is not found.")
    return args


def main(argv):
    simplefilter("ignore", category=ConvergenceWarning)
    random.seed(0)
    print(config)
    args = parse_args(argv)
    # hyper-parameter
    chkpt_dir = os.path.join("/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/models/morphomnist",
                             args.checkpoint_dir)
    
    setup_logging(level="info",
                  logfile=os.path.join(chkpt_dir, "stongencoder.log"))
    weak_size = args.weak_size
    strong_size = args.strong_size
    n_epochs = args.n_epochs
    pond = args.ponderation
    pretrained_path = args.pretrained_path

    print(chkpt_dir)
    os.makedirs(chkpt_dir, exist_ok=True)

    # Instantiate Dataset and Data Loader
    targets = ["label", "area_img", "length", 
               "thickness_img", "slant", "width", "height"]
    train_dataset = MorphoMNISTDataset(split="train", targets=targets)
    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=8)
    
    test_dataset = MorphoMNISTDataset(split="test", targets=targets)
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=8)
    
    # build model
    strong_encoder = StrongEncoder(common_dim=weak_size, strong_dim=strong_size)
    # train model
    strong_encoder.beta = pond
    # strong_encoder.fit(train_loader, test_loader, n_epochs, chkpt_dir, pretrained_path)
    strong_encoder.test(train_loader, test_loader, chkpt_dir=chkpt_dir, epoch=n_epochs)

if __name__ == "__main__":
    main(argv=sys.argv[1:])
