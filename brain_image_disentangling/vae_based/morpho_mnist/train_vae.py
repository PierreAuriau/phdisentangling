# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import itertools

from torch.utils.data import DataLoader

from vae import VAE
from datasets import MorphoMNISTDataset
from loggers import TrainLogger, setup_logging
from config import Config

config = Config()

"""
def gs_beta():
    modality = "photo"
    
    config.path2models = "/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/models/morpho_mnist"
    if modality == "photo":
        chkpt_dir = os.path.join(config.path2models, "20240721_vae-photo")
    elif modality == "sketch":
        chkpt_dir = os.path.join(config.path2models, "20240720_vae-sketch")
    os.makedirs(chkpt_dir, exist_ok=True)
    nb_epochs = 500

    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "logs.log"))

    # data loaders
    train_dataset = PhotoSketchingDataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=64,
                              shuffle=True, num_workers=4)
    
    val_dataset = PhotoSketchingDataset(split="val")
    val_loader = DataLoader(val_dataset, batch_size=64,
                            shuffle=False, num_workers=4)
    
    test_dataset = PhotoSketchingDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=False, num_workers=4)
    
    # model
    input_channels = 3 if modality=="photo" else 1
    latent_dim = "" # FIXME
    nb_layers = "" # FIXME

    for beta in [10, 100, 1000]: # FIXME
        model = VAE(input_channels=input_channels,
                    latent_dim=latent_dim,
                    nb_layers=nb_layers,
                    beta=beta)
        print(f"Beta: {beta}")
        chkpt_dir_gs = os.path.join(chkpt_dir, f"beta_{beta}")
        os.makedirs(chkpt_dir_gs, exist_ok=True)
        model.fit(train_loader, val_loader, chkpt_dir=chkpt_dir_gs, 
                  nb_epochs=nb_epochs, modality=modality, lr=1e-4)
        model.load_chkpt(os.path.join(chkpt_dir_gs, f"vae_mod-{modality}_ep-{nb_epochs-1}.pth"))
        reconstructions = model.get_reconstructions(val_loader, modality)
        filename = f"reconstructions_set-validation_ep-{nb_epochs-1}.npy"
        np.save(os.path.join(chkpt_dir_gs, filename), reconstructions)
    """

def gs_architecture_latent_dim():
    modality = "image"
    config.path2models = "/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/models/morpho_mnist"
    chkpt_dir = os.path.join(config.path2models, "20240903_vae-image")
    os.makedirs(chkpt_dir, exist_ok=True)

    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "logs.log"))
    
    if modality == "skeleton":
        targets = ["label", "fracture_x", "fracture_y", "area_skel", "length_skel", 
                   "thickness_skel", "slant_skel", "width_skel", "height_skel"]
    else:
        targets = ["label", "swelling_amount", "area_img", "length_img", 
                   "thickness_img", "slant_img", "width_img", "height_img"]

    # data loaders
    train_dataset = MorphoMNISTDataset(split="train", targets=targets)
    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=4)
    
    test_dataset = MorphoMNISTDataset(split="test", targets=targets)
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=4)
    
    # model
    nb_epochs = 500

    for latent_dim in [16, 32, 64]:
        for nb_layers in [3]:
            print(f"\nLatent dimension: {latent_dim} - Number of layers: {nb_layers}")
            chkpt_dir_gs = os.path.join(chkpt_dir, f"lt-{latent_dim}_ly-{nb_layers}")
            os.makedirs(chkpt_dir_gs, exist_ok=True)
            model = VAE(latent_dim=latent_dim,
                        nb_layers=nb_layers)
            model.fit(train_loader, chkpt_dir=chkpt_dir_gs, 
                      nb_epochs=nb_epochs, modality=modality, lr=1e-4)
            model.test(test_loader, epoch=nb_epochs-1, 
                       modality=modality, chkpt_dir=chkpt_dir_gs)
            model.test_linear_probe(train_loader, test_loader, epoch=nb_epochs-1, 
                            modality=modality, chkpt_dir=chkpt_dir_gs,
                            targets=targets)
    
def train(chkpt_dir, latent_dim, nb_layers, nb_epochs, 
          reconstruction_loss="mse", beta=1.0):
    
    for modality in ("image", "skeleton"):    
        if modality == "skeleton":
            targets = ["label", "fracture", "area_skel", "length_skel", 
                       "thickness_skel", "slant_skel", "width_skel", "height_skel"]
        else:
            targets = ["label", "thickening_amount", "area_img", "length_img", 
                       "thickness_img", "slant_img", "width_img", "height_img"]

        # data loaders
        train_dataset = MorphoMNISTDataset(split="train", targets=targets)
        train_loader = DataLoader(train_dataset, batch_size=128,
                                shuffle=True, num_workers=4)
        
        test_dataset = MorphoMNISTDataset(split="test", targets=targets)
        test_loader = DataLoader(test_dataset, batch_size=128,
                                shuffle=False, num_workers=4)
        
        # model
        print(f"Model training on {modality}")
        chkpt_dir_mod = os.path.join(chkpt_dir, f"mod-{modality}")
        os.makedirs(chkpt_dir_mod, exist_ok=True)
        model = VAE(latent_dim=latent_dim,
                    nb_layers=nb_layers)
        model.beta = beta
        model.reconstruction_loss = reconstruction_loss
        model.fit(train_loader, chkpt_dir=chkpt_dir_mod, 
                nb_epochs=nb_epochs, modality=modality, lr=1e-4)
        model.test(test_loader, epoch=nb_epochs-1, 
                modality=modality, chkpt_dir=chkpt_dir_mod)
        model.test_linear_probe(train_loader, test_loader, epoch=nb_epochs-1, 
                                modality=modality, chkpt_dir=chkpt_dir_mod,
                                targets=targets)
        outputs = model.get_reconstructions(test_loader, modality=modality, raw=True)
        np.save(os.path.join(chkpt_dir_mod, f"outputs_set-test_ep-{nb_epochs-1}"),
                             outputs)

def gs_loss_latent_dim_pytorchvae():
    
    config.path2models = "/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/models/morpho_mnist"
    chkpt_dir = os.path.join(config.path2models, "20250307_pytorchvae")
    os.makedirs(chkpt_dir, exist_ok=True)

    setup_logging(level="info", 
                logfile=os.path.join(chkpt_dir, "logs.log"))
    
    for latent_dim, loss in itertools.product([64, 128], ["mse", "l1"]):
        chkpt_dir_lt = os.path.join(chkpt_dir, f"lt-{latent_dim}_loss-{loss}")
        os.makedirs(chkpt_dir, exist_ok=True)
        train(chkpt_dir_lt, latent_dim, nb_layers=5, nb_epochs=300, reconstruction_loss=loss)
    



if __name__ == "__main__":

    """modality = "skeleton"
    config.path2models = "/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/models/morpho_mnist"
    chkpt_dir = os.path.join(config.path2models, "20240830_vae-skeleton", "lt-16_ly-3")
    os.makedirs(chkpt_dir, exist_ok=True)

    setup_logging(level="info", 
                  logfile=os.path.join(chkpt_dir, "logs.log"))
    
    if modality == "skeleton":
        targets = ["label", "fracture_x", "fracture_y", "area_skel", "length_skel", 
                   "thickness_skel", "slant_skel", "width_skel", "height_skel"]
    else:
        targets = ["label", "swelling_amount", "area_img", "length_img", 
                   "thickness_img", "slant_img", "width_img", "height_img"]

    # data loaders
    train_dataset = MorphoMNISTDataset(split="train", targets=targets)
    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=4)
    
    test_dataset = MorphoMNISTDataset(split="test", targets=targets)
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=4)
    
    # model
    latent_dim = 16
    nb_layers = 3
    nb_epochs = 500

    model = VAE(latent_dim=latent_dim,
                nb_layers=nb_layers)
    model.fit(train_loader, chkpt_dir=chkpt_dir, 
              nb_epochs=nb_epochs, modality=modality, lr=1e-4)
    model.test(test_loader, epoch=nb_epochs-1, 
               modality=modality, chkpt_dir=chkpt_dir)
    model.test_linear_probe(train_loader, test_loader, epoch=nb_epochs-1, 
                            modality=modality, chkpt_dir=chkpt_dir,
                            targets=["label", "fracture_x", "fracture_y", "area_skel", "length_skel", 
                                     "thickness_skel", "slant_skel", "width_skel", "height_skel"])"""
    
    gs_loss_latent_dim_pytorchvae()
