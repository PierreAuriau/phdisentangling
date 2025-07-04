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
from loss import align_loss, uniform_loss, norm, joint_entropy_loss
import logging
from loggers import TrainLogger
from weak_encoder import WeakEncoder

logging.setLoggerClass(TrainLogger)


class StrongEncoder(nn.Module):
    def __init__(self, common_dim, strong_dim):
        super(StrongEncoder, self).__init__()

        self.common_dim = common_dim
        self.strong_dim = strong_dim

        # encoder
        self.common_enc = resnet18(pretrained=False)
        self.strong_enc = resnet18(pretrained=False)
        self.feature_dim = 512
        
        self.common_enc.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), 
                                           stride=(2, 2), padding=(3, 3), bias=False)
        self.common_enc.fc = nn.Linear(self.feature_dim, common_dim)

        self.strong_enc.conv1 =  nn.Conv2d(1, 64, kernel_size=(7, 7), 
                                           stride=(2, 2), padding=(3, 3), bias=False)
        self.strong_enc.fc = nn.Linear(self.feature_dim, strong_dim)

        # Add MLP projection.
        self.common_projector = nn.Sequential(nn.Linear(common_dim, 128),
                                            nn.BatchNorm1d(128),
                                            nn.ReLU(),
                                            nn.Linear(128, common_dim))

        self.strong_projector = nn.Sequential(nn.Linear(strong_dim, 128),
                                              nn.BatchNorm1d(128),
                                              nn.ReLU(),
                                              nn.Linear(128, strong_dim))
        self.beta = 1.0
        self.logger = logging.getLogger("strongencoder")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.logger.info(f"Device: {self.device}")

    def forward(self, x):
        common_rep = self.common_enc(x)
        strong_rep = self.strong_enc(x)

        common_head = self.common_projector(common_rep)
        strong_head = self.strong_projector(strong_rep)

        return common_rep, common_head, strong_rep, strong_head
    
    def fit(self, train_loader, test_loader, nb_epochs, chkpt_dir, pretrained_path):
        
        # weak encoder
        self.weak_encoder = WeakEncoder(weak_dim=self.common_dim)
        self.weak_encoder.load_chkpt(pretrained_path)
        self.weak_encoder = self.weak_encoder.to(self.device)
        # define optimizer
        self.optimizer = Adam(self.parameters(), lr=3e-4)
        # data augmentation
        data_aug_pipeline = SimCLRDataAugmentation()

        # Train model
        self.logger.reset_history()
        self.logger.info("Epoch : 0")
        for label in test_loader.dataset.targets:
            scores = self.test_linear_probe(train_loader, test_loader, label)
            self.logger.step()
            self.logger.store({"epoch": 0, "set": "test", "label": label,
                              "score_strong": scores["strong"], "score_common": scores["strong_common"]})

        for epoch in range(1, nb_epochs+1):
            self.train()
            self.weak_encoder.eval()
            train_loss = 0
            self.logger.step()
            for batch_idx, batch in enumerate(train_loader):
                with torch.no_grad():
                    weak_view1 = data_aug_pipeline(batch["skeleton"].to(self.device))
                    weak_view2 = data_aug_pipeline(batch["skeleton"].to(self.device))
                    strong_view1 = data_aug_pipeline(batch["image"].to(self.device))
                    strong_view2 = data_aug_pipeline(batch["image"].to(self.device))

                # weak_head
                _, weak_head_1 = self.weak_encoder(weak_view1)
                _, weak_head_2 = self.weak_encoder(weak_view2)

                # strong head
                _, common_strong_head_1, _, strong_head_1 = self(strong_view1)
                _, common_strong_head_2, _, strong_head_2 = self(strong_view2)

                # common strong to weak
                common_strong_align_loss = align_loss(norm(weak_head_1.detach()), norm(common_strong_head_1))
                common_strong_align_loss +=  align_loss(norm(weak_head_2.detach()), norm(common_strong_head_2))
                common_strong_align_loss /= 2.0
                common_strong_uniform_loss = (uniform_loss(norm(common_strong_head_2)) + uniform_loss(norm(common_strong_head_1))) / 2.0
                common_strong_loss = common_strong_align_loss + common_strong_uniform_loss

                # strong loss
                strong_align_loss = align_loss(norm(strong_head_1), norm(strong_head_2))
                strong_uniform_loss = (uniform_loss(norm(strong_head_2)) + uniform_loss(norm(strong_head_1))) / 2.0
                strong_loss = strong_align_loss + strong_uniform_loss

                # mi minimization loss
                jem_loss = joint_entropy_loss(norm(strong_head_1), norm(weak_head_1.detach()))
                jem_loss = jem_loss + joint_entropy_loss(norm(strong_head_2), norm(weak_head_2.detach()))
                jem_loss = jem_loss / 2.0

                loss = strong_loss + common_strong_loss + self.beta*jem_loss

                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.logger.store({"strong_alignement": strong_align_loss.item(),  "strong_uniformity": strong_uniform_loss.item(),
                                   "common_alignement": common_strong_align_loss.item(), "common_uniformity": common_strong_uniform_loss.item(),
                                   "jem": jem_loss.item()})

            self.logger.reduce(reduce_fx="sum")
            self.logger.store({"epoch": epoch, "set": "train", "loss": train_loss})

            if (epoch % 10 == 0) or (epoch == nb_epochs):
                self.logger.info(f"Epoch:  {epoch}")
                for label in batch.keys():
                    if label not in ["image", "skeleton"]:
                        scores = self.test_linear_probe(train_loader, test_loader, label)
                        self.logger.step()
                        self.logger.store({"epoch": epoch, "set": "test", "label": label,
                           "score_strong": scores["strong"], "score_common": scores["strong_common"]})
                self.save_chkpt(filename=os.path.join(chkpt_dir,
                                                      f"strong_encoder_ep-{epoch}.pth"))
            self.logger.save(chkpt_dir, filename="_train")

    def test(self, train_loader, test_loader, chkpt_dir, epoch):
        self.load_chkpt(os.path.join(chkpt_dir, f"strong_encoder_ep-{epoch}.pth"))
        self.logger.info(f"Test model on epoch {epoch}")
        logs = defaultdict(list)
        for label in test_loader.dataset.targets:
            scores = self.test_linear_probe(train_loader, test_loader, label)
            for encoder, score in scores.items():
                logs["epoch"].append(epoch)
                logs["set"].append("test")
                logs["label"].append(label)
                logs["encoder"].append(encoder)
                logs["score"].append(score)
        df = pd.DataFrame(logs)
        filename = os.path.join(chkpt_dir, f"strong_encoder_ep-{epoch}_test.csv")
        if os.path.exists(filename):
            df_pred = pd.read_csv(filename)
            df_to_save = pd.concat((df_pred, df), axis=0)
            df_to_save.to_csv(os.path.join(chkpt_dir, f"strong_encoder_test.csv"))
        else:
            df.to_csv(os.path.join(chkpt_dir, f"strong_encoder_test.csv"))
    
    # test linear probes
    def test_linear_probe(self, train_loader, test_loader, label):
        self.eval()
        with torch.no_grad():
            X_train_strong = []
            X_test_strong = []
            X_train_strong_common = []
            X_test_strong_common = []
            y_train = []
            y_test = []
            for batch in train_loader:
                common_strong_rep, _, strong_rep, _ = self(batch["image"].to(self.device))
                X_train_strong.extend(strong_rep.cpu().numpy())
                X_train_strong_common.extend(common_strong_rep.cpu().numpy())
                y_train.extend(batch[label].cpu().numpy())
            for batch in test_loader:
                common_strong_rep, _, strong_rep, _ = self(batch["image"].to(self.device))
                X_test_strong.extend(strong_rep.cpu().numpy())
                X_test_strong_common.extend(common_strong_rep.cpu().numpy())
                y_test.extend(batch[label].cpu().numpy())
            X_train_strong = np.array(X_train_strong)
            X_test_strong = np.array(X_test_strong)
            X_train_strong_common = np.array(X_train_strong_common)
            X_test_strong_common = np.array(X_test_strong_common)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

        scores = {}
        if label in ["digit", "fracture"]:
            log_reg = LogisticRegression().fit(X_train_strong, y_train)
            if label == "fracture":
                log_reg_score = roc_auc_score(y_score=log_reg.predict_proba(X_test_strong)[:, 1],
                                              y_true=y_test)
            else:
                log_reg_score = log_reg.score(X_test_strong, y_test)
            self.logger.info(f"strong evaluated on {label}: {log_reg_score:.2f}")
            scores["strong"] = log_reg_score
            log_reg = LogisticRegression().fit(X_train_strong_common, y_train)
            log_reg_score = log_reg.score(X_test_strong_common, y_test)
            self.logger.info(f"strong common evaluated on {label}: {log_reg_score:.2f}")
            scores["strong_common"] = log_reg_score
        else:
            ridge = Ridge().fit(X_train_strong, y_train)
            ridge_score = ridge.score(X_test_strong, y_test)
            self.logger.info(f"strong evaluated on {label}: {ridge_score:.2f}")
            scores["strong"] = ridge_score
            ridge = Ridge().fit(X_train_strong_common, y_train)
            ridge_score = ridge.score(X_test_strong_common, y_test)
            self.logger.info(f"strong common evaluated on {label}: {ridge_score:.2f}")
            scores["strong_common"] = ridge_score

        return scores

    def save_chkpt(self, filename):
        torch.save({"strong": {"encoder": self.strong_enc.state_dict(),
                               "projector": self.strong_projector.state_dict()},
                    "common": {"encoder": self.common_enc.state_dict(),
                               "projector": self.common_projector.state_dict()}},
                   filename)
    
    def load_chkpt(self, filename):
        chkpt = torch.load(filename)
        status = self.strong_enc.load_state_dict(chkpt["strong"]["encoder"], strict=False)
        self.logger.info(f"Loading strong encoder: {status}")
        status = self.common_enc.load_state_dict(chkpt["common"]["encoder"], strict=False)
        self.logger.info(f"Loading common encoder: {status}")
        self.strong_projector.load_state_dict(chkpt["strong"]["projector"], strict=False)
        self.common_projector.load_state_dict(chkpt["common"]["projector"], strict=False)
