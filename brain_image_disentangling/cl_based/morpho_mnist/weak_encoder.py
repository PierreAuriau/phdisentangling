# !/usr/bin/env python
# -*- coding: utf-8 -*-

# import function
import os
from collections import defaultdict
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.optim import Adam
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from augmentation import SimCLRDataAugmentation
from loss import align_loss, uniform_loss, norm
from tqdm import tqdm
import logging
from loggers import TrainLogger

logging.setLoggerClass(TrainLogger)

class WeakEncoder(nn.Module):
    def __init__(self, weak_dim):
        super(WeakEncoder, self).__init__()

        self.weak_dim = weak_dim

        # encoder
        self.weak_enc = resnet18(pretrained=False)
        self.feature_dim = 512
        self.weak_enc.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), 
                                        stride=(2, 2), padding=(3, 3), bias=False)
        self.weak_enc.fc = nn.Linear(self.feature_dim, weak_dim)    

        # Add MLP projection.
        self.weak_projector = nn.Sequential(nn.Linear(weak_dim, 128),
                                            nn.BatchNorm1d(128),
                                            nn.ReLU(),
                                            nn.Linear(128, weak_dim))
        self.logger = logging.getLogger("weakencoder")
        
        self.tau = 1.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, x):
        weak_rep = self.weak_enc(x)

        weak_head = self.weak_projector(weak_rep)

        return weak_rep, weak_head
    
    # train encoder
    def fit(self, train_loader, test_loader, n_epochs, chkpt_dir):
        # define optimizer
        optimizer = Adam(list(self.parameters()), lr=3e-4)
        # data augmentation
        data_aug_pipeline = SimCLRDataAugmentation()

        # train model
        self.cuda()
        self.logger.reset_history()
        for epoch in tqdm(range(n_epochs), desc="epochs"):
            self.train()
            train_loss = 0
            train_align = 0
            train_uniform = 0
            self.logger.step()
            for batch_idx, batch in enumerate(train_loader):
                with torch.no_grad():
                    weak_view1 = data_aug_pipeline(batch["skeleton"])
                    weak_view2 = data_aug_pipeline(batch["skeleton"])

                # weak_head
                _, weak_head_1 = self(weak_view1.cuda())
                _, weak_head_2 = self(weak_view2.cuda())

                # weak loss
                weak_align_loss = align_loss(norm(weak_head_1), norm(weak_head_2))
                weak_uniform_loss = (uniform_loss(norm(weak_head_2)) + uniform_loss(norm(weak_head_1))) / 2.0
                weak_loss = weak_align_loss + self.tau * weak_uniform_loss

                loss = weak_loss
                
                loss.backward()
                train_loss += loss.item()
                train_align += weak_align_loss.item()
                train_uniform += weak_uniform_loss.item()
                optimizer.step()
                optimizer.zero_grad()
            self.logger.reduce(reduce_fx="sum")
            self.logger.store({"epoch": epoch, "set": "train", "loss": train_loss,
                               "alignement": train_align, "uniformity": train_uniform})

            if (epoch % 10 == 0) or (epoch == n_epochs-1):
                self.logger.info(f"\nepoch: {epoch}\n" + "-"*(len(str(epoch))+7))
                self.logger.info(f"train loss: {train_loss:.2e} (align={train_align:.2e}, uniform={train_uniform:.2e})")
                for label in batch.keys():
                    if label not in ("skeleton", "image"):
                        score = self.test_linear_probe(train_loader, test_loader, label)
                        self.logger.step()
                        self.logger.store({"epoch": epoch, "set": "test", "label": label, "score": score})
                self.save_chkpt(filename=os.path.join(chkpt_dir,
                                                      f"weak_encoder_ep-{epoch}.pth"))
            self.logger.save(chkpt_dir, filename="_train")

    def test(self, train_loader, test_loader, epoch, chkpt_dir):
        self.logger.info(f"Test model on epoch {epoch}")
        self.load_chkpt(os.path.join(chkpt_dir, f"weak_encoder_ep-{epoch}.pth"))
        logs = defaultdict(list)
        for label in test_loader.dataset.targets:
            score = self.test_linear_probe(train_loader, test_loader, label)
            logs["epoch"].append(epoch)
            logs["set"].append("test")
            logs["label"].append(label)
            logs["score"].append(score)
        df = pd.DataFrame(logs)
        df.to_csv(os.path.join(chkpt_dir, f"weak_encoder_ep-{epoch}_test.csv"))


    # test linear probes
    def test_linear_probe(self, train_loader, test_loader, label):
        self.eval()
        with torch.no_grad():
            X_train_weak = []
            X_test_weak = []
            y_weak_train = []
            y_weak_test = []
            for batch in train_loader:
                weak_rep, _ = self(batch["skeleton"].cuda())
                X_train_weak.extend(weak_rep.cpu().numpy())
                y_weak_train.extend(batch[label].cpu().numpy())
            for batch in test_loader:
                weak_rep, _ = self(batch["skeleton"].cuda())
                X_test_weak.extend(weak_rep.cpu().numpy())
                y_weak_test.extend(batch[label].cpu().numpy())
            X_train_weak = np.array(X_train_weak)
            X_test_weak = np.array(X_test_weak)
            y_weak_train = np.array(y_weak_train)
            y_weak_test = np.array(y_weak_test)

        if label in ["digit", "fracture"]:
            log_reg = LogisticRegression().fit(X_train_weak, y_weak_train)
            if label == "fracture":
                score = roc_auc_score(y_score=log_reg.predict_proba(X_test_weak)[:, 1],
                                      y_true=y_weak_test)
            else:
                score = log_reg.score(X_test_weak, y_weak_test) 
        else:
            ridge = Ridge().fit(X_train_weak, y_weak_train)
            score = ridge.score(X_test_weak, y_weak_test)
        self.logger.info(f"weak trained on {label}: {score:.2f}")
        return score

    def save_chkpt(self, filename):
        torch.save({"encoder": self.weak_enc.state_dict(),
                    "projector": self.weak_projector.state_dict()},
                   filename)

    def load_chkpt(self, filename):
        chkpt = torch.load(filename)
        status = self.weak_enc.load_state_dict(chkpt["encoder"], strict=False)
        print(f"Loading encoder: {status}")
        status = self.weak_projector.load_state_dict(chkpt["projector"], strict=False)
        print(f"Loading projector: {status}")
