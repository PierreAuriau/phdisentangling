# !/usr/bin/env python
# -*- coding: utf-8 -*-

# import functions
import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms.transforms import Compose
from torch.cuda.amp import GradScaler, autocast

# project import
from loss import align_loss, uniform_loss, norm
from datamanager import ClinicalDataManager
from utils import save_checkpoint, setup_logging, save_hyperparameters
from da_module import DAModule, ToArray, ToTensor
from encoder import Encoder

logger = logging.getLogger()


def weak_loss(z_1, z_2):
    weak_align_loss = align_loss(norm(z_1), norm(z_2))
    weak_uniform_loss = (uniform_loss(norm(z_1)) + uniform_loss(norm(z_2))) / 2.0
    return weak_align_loss + weak_uniform_loss


# train encoder
def train(weak_encoder, manager, nb_epochs, checkpointdir, exp_name):
    # define optimizer
    optimizer = Adam(list(weak_encoder.parameters()), lr=1e-4)
    # data augmentation
    da_pipeline = Compose([ToArray(), DAModule(("Cutout", )), ToTensor()])

    nb_epochs_per_saving = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device {device}")
    
    scaler = GradScaler()
    
    history = {"epoch": [],
               "train_loss": [],
               "val_loss": []}
    loader = manager.get_dataloader(train=True,
                                    validation=True)
    # train model
    weak_encoder = weak_encoder.to(device)
    nb_batch = len(loader.train) + len(loader.validation)
    
    for epoch in range(nb_epochs):
        pbar = tqdm(total=nb_batch, desc=f"Epoch {epoch}")
        
        # train
        weak_encoder.train()
        train_loss = 0
        for dataitem in loader.train:
            pbar.update()
            inputs = dataitem.inputs
            weak_view1 = da_pipeline(inputs)
            weak_view2 = da_pipeline(inputs)
            
            optimizer.zero_grad()
            with autocast():

                _, weak_head_1 = weak_encoder(weak_view1.to(device))
                _, weak_head_2 = weak_encoder(weak_view2.to(device))

                # weak loss
                loss = weak_loss(weak_head_1, weak_head_2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
        
        # eval
        weak_encoder.eval()
        val_loss = 0
        for dataitem in loader.validation:
            pbar.update()
            with torch.no_grad():
                weak_view1 = da_pipeline(dataitem.inputs)
                weak_view2 = da_pipeline(dataitem.inputs)
                _, weak_head_1 = weak_encoder(weak_view1.to(device))
                _, weak_head_2 = weak_encoder(weak_view2.to(device))

                # weak loss
                loss = weak_loss(weak_head_1, weak_head_2)

            val_loss += loss.item()
        pbar.close()
        logger.info(f"Epoch: {epoch} | Train loss: {train_loss:.2f} | Val loss: {val_loss:.2f}")
        # saving
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if (epoch % nb_epochs_per_saving == 0 or epoch == nb_epochs-1) and epoch > 0:
            save_checkpoint(
                model=weak_encoder.state_dict(),
                epoch=epoch,
                outdir=checkpointdir,
                name=exp_name,
                optimizer=optimizer.state_dict())
            
            df = pd.DataFrame(history)
            df.to_csv(os.path.join(checkpointdir, f"exp-{exp_name}_ep-{epoch}_losses.csv"), index=False)

def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description='Train weak encoder')
    parser.add_argument(
        "-c", "--checkpoint_dir", type=str, required=True,
        help="Directory where models are saved")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, choices=["asd", "bd", "scz"])
    parser.add_argument(
        "-e", "--exp_name", type=str, required=True)
    parser.add_argument(
        "-l", "--latent_dim", type=int, default=32,
        help="Latent space size of the weak encoder. Default is 32.")
    parser.add_argument(
        "-n", "--nb_epochs", type=int, default=50,
        help="Number of epochs for training. Default is 50.")
    args = parser.parse_args(argv)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        print("Checkpoint directory created.")
    return args


def main(argv):
    args = parse_args(argv)

    setup_logging(logfile=os.path.join(args.checkpoint_dir, 
                                       f"exp-{args.exp_name}.log"))
    
    save_hyperparameters(args)
    
    # Instantiate datamanager
    manager = ClinicalDataManager(root="/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/data/root", 
                                  db=args.dataset, preproc="skeleton", labels="diagnosis", 
                                  sampler="random", batch_size=32, 
                                  num_workers=8, pin_memory=True)

    # build model
    weak_encoder = Encoder(backbone="resnet18", n_embedding=args.latent_dim)

    
    # train model
    train(weak_encoder=weak_encoder, manager=manager, nb_epochs=args.nb_epochs, 
          exp_name=args.exp_name, checkpointdir=args.checkpoint_dir)


if __name__ == "__main__":
    main(argv=sys.argv[1:])
    