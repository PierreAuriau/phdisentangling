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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# project import
from datamanager import ClinicalDataManager
from utils import setup_logging, get_chk_name
from encoder import Encoder

logger = logging.getLogger()


# train encoder
def test(weak_encoder, manager, list_epochs, checkpointdir, exp_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device {device}")
    
    history = {"epoch": [],
               "train": [],
               "val": [],
               "test_ext": [],
               "test_intra": []}
    
    # test model
    weak_encoder = weak_encoder.to(device)

    for epoch in list_epochs:
        logger.info(f"Epoch : {epoch}")

        # Load model
        logger.info(f"Loading model...")
        checkpoint = torch.load(os.path.join(checkpointdir, get_chk_name(exp_name=exp_name, epoch=epoch)))
        status = weak_encoder.load_state_dict(checkpoint["model"], strict=False)
        logger.info(f"Loading info : {status}")

        # get embeddings
        weak_encoder.eval()
        # training set
        loader = manager.get_dataloader(train=True,
                                        validation=True)
        pbar = tqdm(total=len(loader.train), desc=f"Training")
        repr_train = []
        label_train = []
        for dataitem in loader.train:
            pbar.update()
            with torch.no_grad():
                inputs = dataitem.inputs
                weak_repr, _ = weak_encoder(inputs.to(device))
            repr_train.extend(weak_repr.detach().cpu().numpy())
            label_train.extend(dataitem.labels.detach().cpu().numpy())
        repr_train = np.asarray(repr_train)
        label_train = np.asarray(label_train)
        pbar.close()
        # validation set
        pbar = tqdm(total=len(loader.validation), desc=f"Validation")
        repr_val = []
        label_val = []
        for dataitem in loader.validation:
            pbar.update()
            with torch.no_grad():
                inputs = dataitem.inputs
                weak_repr, _ = weak_encoder(inputs.to(device))
            repr_val.extend(weak_repr.detach().cpu().numpy())
            label_val.extend(dataitem.labels.detach().cpu().numpy())
        repr_val = np.asarray(repr_val)
        label_val = np.asarray(label_val)
        pbar.close()
        # internal test
        loader = manager.get_dataloader(test_intra=True)
        pbar = tqdm(total=len(loader.test), desc=f"Internal Test")
        repr_test_intra = []
        label_test_intra = []
        for dataitem in loader.test:
            pbar.update()
            with torch.no_grad():
                inputs = dataitem.inputs
                weak_repr, _ = weak_encoder(inputs.to(device))
            repr_test_intra.extend(weak_repr.detach().cpu().numpy())
            label_test_intra.extend(dataitem.labels.detach().cpu().numpy())
        repr_test_intra = np.asarray(repr_test_intra)
        label_test_intra = np.asarray(label_test_intra)
        pbar.close()
        # external test
        loader = manager.get_dataloader(test=True)
        pbar = tqdm(total=len(loader.test), desc=f"External Test")
        repr_test_ext = []
        label_test_ext = []
        for dataitem in loader.test:
            pbar.update()
            with torch.no_grad():
                inputs = dataitem.inputs
                weak_repr, _ = weak_encoder(inputs.to(device))
            repr_test_ext.extend(weak_repr.detach().cpu().numpy())
            label_test_ext.extend(dataitem.labels.detach().cpu().numpy())
        repr_test_ext = np.asarray(repr_test_ext)
        label_test_ext = np.asarray(label_test_ext)
        pbar.close()

        logger.info("Training Logistic Regression")
        clf = LogisticRegression()
        clf = clf.fit(repr_train, label_train)
        y_pred_train = clf.predict(repr_train)
        y_pred_val = clf.predict(repr_val)
        y_pred_test_intra = clf.predict(repr_test_intra)
        y_pred_test_ext = clf.predict(repr_test_ext)

        roc_auc_train = roc_auc_score(y_score=y_pred_train, y_true=label_train)
        roc_auc_val = roc_auc_score(y_score=y_pred_val, y_true=label_val)
        roc_auc_test_intra = roc_auc_score(y_score=y_pred_test_intra, y_true=label_test_intra)
        roc_auc_test_extra = roc_auc_score(y_score=y_pred_test_ext, y_true=label_test_ext)

        logger.info(f"ROC AUC: Train {roc_auc_train} | Val {roc_auc_val} | Internal Test {roc_auc_test_intra} | External Test {roc_auc_test_extra}")

        # saving
        history["epoch"].append(epoch)
        history["train"].append(roc_auc_train)
        history["val"].append(roc_auc_val)
        history["test_intra"].append(roc_auc_test_intra)
        history["test_ext"].append(roc_auc_test_extra)
        
        df = pd.DataFrame(history)
    df.to_csv(os.path.join(checkpointdir, f"Test_exp-{exp_name}_rocauc.csv"), index=False)


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
        "-n", "--list_epochs", type=int, default=49, nargs="+",
        help="Epochs to test")
    args = parser.parse_args(argv)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        print("Checkpoint directory created.")
    return args


def main(argv):
    args = parse_args(argv)

    setup_logging(logfile=os.path.join(args.checkpoint_dir, 
                                       f"test_exp-{args.exp_name}.log"))
        
    # Instantiate datamanager
    manager = ClinicalDataManager(root="/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/data/root", 
                                  db=args.dataset, preproc="skeleton", labels="diagnosis", 
                                  sampler="random", batch_size=32, 
                                  num_workers=8, pin_memory=True)

    # build model
    weak_encoder = Encoder(backbone="resnet18", n_embedding=args.latent_dim)
        
    # train model
    test(weak_encoder=weak_encoder, manager=manager, list_epochs=args.list_epochs, 
          exp_name=args.exp_name, checkpointdir=args.checkpoint_dir)


if __name__ == "__main__":
    main(argv=sys.argv[1:])
    