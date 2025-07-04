# !/usr/bin/env python
# -*- coding: utf-8 -*-

# import functions
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
# ignore warning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import random
# project import
from dataset import MorphoMNISTDataset
from weak_encoder import WeakEncoder
from loggers import setup_logging

from config import Config

config = Config()

def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description='Train weak encoder')
    parser.add_argument(
        "-c", "--checkpoint_dir", type=str, required=True,
        help="Directory where models are saved")
    parser.add_argument(
        "-w", "--weak_size", type=int, default=32,
        help="Latent space size of the weak encoder. Default is 32.")
    parser.add_argument(
        "-n", "--n_epochs", type=int, default=250,
        help="Number of epochs for training. Default is 250.")
    args = parser.parse_args(argv)
    #if not os.path.exists(args.checkpoint_dir):
    #    os.makedirs(args.checkpoint_dir)
    #    print("Checkpoint directory created.")
    return args


def main(argv):
    simplefilter("ignore", category=ConvergenceWarning)
    random.seed(0)

    args = parse_args(argv)

    # hyper-parameter
    weak_size = args.weak_size
    n_epochs = args.n_epochs
    chkpt_dir = os.path.join(config.path_to_models,
                             args.checkpoint_dir)
    setup_logging(level="info")
    print(chkpt_dir)
    os.makedirs(chkpt_dir, exist_ok=True)
    # Instantiate dataset and dataloader
    targets = ["label", "fracture", "area_skel", "length_skel", 
                "thickness_skel", "slant_skel", "width_skel", "height_skel"]
    train_dataset = MorphoMNISTDataset(split="train", targets=targets)
    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=8)
    
    test_dataset = MorphoMNISTDataset(split="test", targets=targets)
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=8)
    
    # build model
    weak_encoder = WeakEncoder(weak_dim=weak_size)

    # train model
    # weak_encoder.fit(train_loader, test_loader, n_epochs, chkpt_dir)

    # save model
    # print("Saving model...")
    # torch.save(weak_encoder, os.path.join(args.checkpoint_dir,
    #                                      f"sep_mod_weak_{weak_modality}.pth"))

    weak_encoder.test(train_loader, test_loader, 
                      epoch=n_epochs-1, chkpt_dir=chkpt_dir)


if __name__ == "__main__":
    main(argv=sys.argv[1:])
