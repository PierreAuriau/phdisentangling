#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed May 18 17:23:21 2022

"""

import json
import os
#import argparse
from config import Config
#from engine import Engine
#from metrics import METRICS
import logging
import logging.config
import datetime
import pickle

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from dl_model import DLModel
from densenet import densenet121
from dataset import MRIDataset
from torch.utils.data import DataLoader, RandomSampler

from sklearn.model_selection import StratifiedKFold

from dl_training.datamanager import ClinicalDataManager


class Args():
    def __init__(self, dico):
        for k, v in dico.items():
            setattr(self, k, v)


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()

    # parser.add_argument("--train_data_path", type=str)
    # parser.add_argument("--train_label_path", type=str)
    # parser.add_argument("--val_data_path", type=str)
    # parser.add_argument("--val_label_path", type=str)
    # parser.add_argument("--test_data_path", type=str)
    # parser.add_argument("--test_label_path", type=str)
    # parser.add_argument("--model_path", type=str)
    # parser.add_argument("--checkpoint_dir", type=str, required=True)
    # parser.add_argument("--exp_name", type=str, required=True)
    # parser.add_argument("--metrics", nargs='+', type=str, choices=list(METRICS.keys()),
    #                     help="Metrics to be computed on validation/test set")
    # #parser.add_argument("--labels", nargs='+', type=str, help="Label(s) to be predicted")
    # parser.add_argument("--train", action="store_true")
    # parser.add_argument("--test", action="store_true")

    # args = parser.parse_args()

    #optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=config.gamma_scheduler, step_size=config.step_size_scheduler)
    #engine = Engine(args, config)
    #engine.run(training=args.train, testing=args.test)

    dico_args = json.load(open('args.json', 'r'))

    args = Args(dico_args)
    config = Config()

    # Create saving directory
    if args.train:
        def to_str(entier):
            return '0' + str(entier) if entier < 10 else str(entier)
        today = datetime.date.today()
        dir_name = str(today.year) + to_str(today.month) + \
            to_str(today.day) + '_' + args.exp_name
        saving_dir = os.path.join(args.checkpoint_dir, dir_name)
        os.makedirs(saving_dir, exist_ok=True)

    else:
        saving_dir = args.model_path

    ## Logging ##
    # console and file handlers
    log_config = json.load(open('logging.json', 'r'))
    if args.train:
        log_config['handlers']['file']['filename'] = os.path.join(
            saving_dir, 'train.log')
    else:
        log_config['handlers']['file']['filename'] = os.path.join(
            saving_dir, 'test.log')

    logging.config.dictConfig(log_config)
    logger = logging.getLogger("main")

    if not args.train and not args.test:
        args.train = True
        logger.info("No mode specify: training mode is set automatically")

    # Hyper-Parameters
    n_folds = 3
    random_state = 0

    net = densenet121(num_classes=config.num_classes, in_channels=1)
    pb='scz'
    device = ('cuda' if config.cuda else 'cpu')
    pos_weights = {"scz": 1.131, "asd": 1.584, "bipolar": 1.584, "sex": 1.0}
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights[pb], dtype=torch.float32,
                                                                device=device))
    scheduler = 1

    # Saving Hyperparameters
    dico_hpp = dico_args.copy()
    dico_hpp.update(config.__dict__)
    dico_hpp['n_folds'] = n_folds
    dico_hpp['random_state'] = random_state
    # dico_hpp['loss'] = 'bce'
    # dico_hpp['net'] = 'densenet'
    # dico_hpp['scheduler'] = 'stepLR'
    # dico_hpp['optimizer'] = 'Adam'
    if args.train:
        with open(os.path.join(saving_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(dico_hpp, f)

    ## Training ##
    if args.train:
        # Loading data
        if args.train:
            assert(args.train_data_path is not None)
            assert(args.train_label_path is not None)
            data = np.load(args.train_data_path)
            labels = pd.read_csv(args.train_label_path)
            if labels['diagnosis'].dtype != 'int64':
                labels['diagnosis'] = np.array(
                    labels['diagnosis'] == 'control', dtype=int)
            labels_arr = labels['diagnosis'].values

        # Cross-Validation

        sampler = 'random'
        root_dir = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/'
        manager = ClinicalDataManager(root=root_dir, preproc='vbm', db='scz', labels=['diagnosis'],
                                      sampler=sampler, batch_size=config.batch_size,
                                      residualize=None, number_of_folds=n_folds, N_train_max=None, device=device,
                                      num_workers=config.num_cpu_workers, pin_memory=True, drop_last=False)
        
        for i in range(n_folds):

            logger.info('# Cross Validation ' + str(i) + ' #')

            # Saving directories
            fold_dir = os.path.join(saving_dir, 'fold_' + str(i))
            os.makedirs(fold_dir, exist_ok=True)

            loader = manager.get_dataloader(
                train=True, validation=True, fold_index=i)

            # Loading model
            model = DLModel(net, loss, config, args,
                            loader_train=loader.train,
                            loader_val=loader.validation,
                            scheduler=scheduler, log_dir=fold_dir)

            model.training()
            path_to_save = os.path.join(fold_dir, 'model.pt')
            model.save_model(path_to_save)

    ## Test ##
    if args.test:
        assert(args.test_data_path is not None)
        assert(args.test_label_path is not None)

        # Load data
        model_path = args.model_path

        for i in range(n_folds):
            loader_test = manager.get_dataloader(test_intra=True, fold_index=i)
            args.model_path = os.path.join(
                model_path, 'fold_'+str(i), 'model.pt')
            fold_dir = os.path.join(model_path, 'fold_' + str(i))

            logger.info("Testing on Internal Test Set")
            # Loading model
            model = DLModel(net, loss, config, args,
                            loader_test=loader_test,
                            scheduler=scheduler,
                            log_dir=fold_dir)

            model.testing(name="_intra")

            logger.info("Testing on External Test Set")
            loader_test = manager.get_dataloader(test_intra=True, fold_index=i)
            # Loading model
            model = DLModel(net, loss, config, args,
                            loader_test=loader_test,
                            scheduler=scheduler,
                            log_dir=fold_dir)

            model.testing(name="_extra")
