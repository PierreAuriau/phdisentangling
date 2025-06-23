# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, \
                            mean_squared_error, r2_score, roc_auc_score

from densenet import densenet121
from resnet import resnet18
from alexnet import alexnet
from mlp import Classifier, Regressor
from log import TrainLogger

logging.setLoggerClass(TrainLogger)


class DLModel(nn.Module):

    def __init__(self, architecture="densenet", n_embedding=256):
        super().__init__()
        
        self.n_embedding = n_embedding

        if architecture == "alexnet":
            self.encoder = alexnet(n_embedding=128, in_channels=1)
            self.classifer = Regressor(latent_dim=128, num_classes=1)
        elif architecture == "resnet":
            self.encoder = resnet18(n_embedding=n_embedding, in_channels=1)
            self.classifier = Classifier(latent_dim=n_embedding,
                                         activation="sigmoid")
        else:
            self.encoder = densenet121(n_embedding=n_embedding, in_channels=1)
            self.classifier = Classifier(latent_dim=n_embedding,
                                         activation="sigmoid")
        
        self.logger = logging.getLogger("classifier")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device used : {self.device}")

        self = self.to(self.device)

    def forward(self, x):
        z = self.encoder(x)
        out = self.classifier(z)
        return out
    
    def configure_optimizers(self, **kwargs):
        return optim.Adam(self.parameters(), **kwargs)

    def fit(self, train_loader, val_loader, nb_epochs, 
            chkpt_dir, pos_weight=1.0, logs={}, **kwargs_optimizer):
        
        self.save_hyperparameters(chkpt_dir, {"nb_epochs": nb_epochs,
                                              "pos_weight": pos_weight,
                                              **kwargs_optimizer})
        self.optimizer = self.configure_optimizers(**kwargs_optimizer)
        self.lr_scheduler = None
        self.scaler = GradScaler()
        self.logger.reset_history()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, 
                                                           dtype=torch.float32,
                                                           device=self.device))
        for epoch in range(nb_epochs):
            self.logger.info(f"Epoch: {epoch}")
            # Training
            train_loss = 0
            self.train()
            self.logger.step()
            for batch in tqdm(train_loader, desc="train"):
                input = batch["input"].to(self.device)
                label = batch["label"].to(self.device)
                loss = self.training_step(input, label)
                train_loss += loss
            self.logger.reduce(reduce_fx="sum")
            self.logger.store({"epoch": epoch, "set": "train", "loss": train_loss, **logs})
            
            # Validation
            if val_loader is not None:
                val_loss = 0
                self.classifier.eval()
                self.encoder.eval()
                self.logger.step()
                for batch in tqdm(val_loader, desc="Validation"):
                    input = batch["input"].to(self.device)
                    label = batch["label"].to(self.device)
                    loss = self.valid_step(input, label)
                    val_loss += loss
                self.logger.reduce(reduce_fx="sum")
                self.logger.store({"epoch": epoch, "set": "validation", "loss": val_loss, **logs})

            if epoch % 10 == 0:
                self.logger.info(f"Loss: train: {train_loss:.2g} / val: {val_loss:.2g}")
                self.logger.info(f"Training duration: {self.logger.get_duration()}")
                self.save_chkpt(chkpt_dir=chkpt_dir, 
                                filename=f'classifier_ep-{epoch}.pth')
        
        self.logger.save(chkpt_dir, filename="_train")
        self.save_chkpt(chkpt_dir=chkpt_dir, 
                        filename=f'classifier_ep-{epoch}.pth')
        self.logger.info(f"End of training: {self.logger.get_duration()}")
    
    def training_step(self, input, label):
        self.optimizer.zero_grad()
        with autocast(device_type=self.device.type, dtype=torch.float16):
            z = self.encoder(input)
            pred = self.classifier(z, return_logits=True)
            loss = self.loss_fn(pred.squeeze(), label)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return loss.item()

    def valid_step(self, input, label):
        with torch.no_grad():
            z = self.encoder(input)
            pred = self.classifier(z, return_logits=True)
            loss = self.loss_fn(pred.squeeze(), label)
        return loss.item()
    
    def test(self, loaders, splits,
             chkpt_dir, epoch, metrics, logs={},
             return_logs=False):
        self.logger.info("Testing predictor")
        self.metrics = metrics
        self.logger.reset_history()
        self.load_chkpt(chkpt_dir=chkpt_dir,
                        filename=f"classifier_ep-{epoch}.pth")
        self.eval()
        for split, loader in zip(splits, loaders):
            self.logger.step()
            values = self.test_step(loader=loader,
                                                split=split)
            self.logger.store({"epoch": epoch,
                               "set": split,
                               **values,
                               **logs})
        if return_logs:
            return self.logger.to_dict()
        self.logger.save(chkpt_dir, filename="_test")
    
    def test_step(self, loader, split):
        y_true = []
        y_pred = []
        for batch in tqdm(loader, desc=split):
            input = batch["input"].to(self.device)
            with torch.no_grad():
                z = self.encoder(input)
                pred = self.classifier(z)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(batch["label"].numpy())
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        values = {}
        for name, metric in self.metrics.items():
            values[name] = metric(y_true=y_true,
                                  y_pred=y_pred)
        return values
        
    def save_hyperparameters(self, chkpt_dir, hp={}):
        hp = {"n_embedding": self.n_embedding, **hp}
        with open(os.path.join(chkpt_dir, "hyperparameters.json"), "w") as f:
            json.dump(hp, f)

    def save_chkpt(self, chkpt_dir, filename, save_optimizer=False):
        to_save = {"encoder": self.encoder.state_dict()}
        if self.classifier is not None:
            to_save["classifier"] = self.classifier.state_dict()
        if self.projector is not None:
            to_save["projector"] = self.projector.state_dict()
        torch.save(to_save,
                   os.path.join(chkpt_dir, filename))
        if save_optimizer:
            torch.save(self.optimizer.state_dict(),
                       os.path.join(chkpt_dir, "optimizer.pth"))
    
    def load_chkpt(self, chkpt_dir, filename, load_optimizer=False):
        chkpt = torch.load(os.path.join(chkpt_dir, filename), weights_only=True)
        status = self.encoder.load_state_dict(chkpt["encoder"], strict=False)
        self.logger.info(f"Loading encoder : {status}")
        status = self.classifier.load_state_dict(chkpt["classifier"], strict=False)
        self.logger.info(f"Loading classifier : {status}")
        if load_optimizer:
            self.optimizer.load_state_dict(torch.load(os.path.join(chkpt_dir, "optimizer.pth"),
                                                      weights_only=True))

if __name__ == "__main__":
    model = DLModel()
    