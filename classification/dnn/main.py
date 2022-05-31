#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:23:21 2022

@source : https://github.com/Duplums/bhb10k-dl-benchmark/blob/main/main.py
"""

import json
import os
#import argparse
from config import Config
#from engine import Engine
#from metrics import METRICS
import logging

#import torch
import torch.nn as nn

from dl_model import DLModel
from densenet import densenet121
from dataset import MRIDataset
from torch.utils.data import DataLoader, RandomSampler

class Args() :
    def __init__(self, train_data_path, train_label_path, val_data_path, val_label_path, test_data_path, test_label_path, 
                 model_path, checkpoint_dir, exp_name, metrics, labels, train, test) :
        
        self.train_data_path = train_data_path
        self.train_label_path = train_label_path
        self.val_data_path = val_data_path
        self.val_label_path = val_label_path
        self.test_data_path = test_data_path
        self.test_label_path = test_label_path
        self.model_path = model_path
        self.checkpoint_dir = checkpoint_dir
        self.exp_name = exp_name
        self.metrics = metrics
        self.train = train
        self.test = test
        self.labels = labels

if __name__=="__main__":

    logger = logging.getLogger("pynet")

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
    train_data_path = dico_args['train_data_path']
    train_label_path = dico_args['train_label_path']
    val_data_path = dico_args['val_data_path']
    val_label_path = dico_args['val_label_path']
    test_data_path = dico_args['test_data_path']
    test_label_path = dico_args['test_label_path']
    model_path = dico_args['model_path']
    checkpoint_dir = dico_args['checkpoint_dir']
    exp_name = dico_args['exp_name']
    metrics = dico_args['metrics']
    labels = dico_args['labels']
    train = dico_args['train']
    test = dico_args['test']
    
    
    args = Args(train_data_path, train_label_path, val_data_path, val_label_path, test_data_path, test_label_path, 
                 model_path, checkpoint_dir, exp_name, metrics, labels, train, test)
    config = Config()
    
    #File creation
    os.makedirs(os.path.join(args.checkpoint_dir, args.exp_name), exist_ok=True)
    log_dir = os.path.join(args.checkpoint_dir, args.exp_name, 'tensorboard')
    os.makedirs(log_dir, exist_ok=True)

    if not args.train and not args.test:
        args.train = True
        logger.info("No mode specify: training mode is set automatically")

    net = densenet121(num_classes=config.num_classes)
    loss = nn.BCEWithLogitsLoss()
    scheduler = 1
    
    loader_train, loader_val, loader_test = (None, None, None)
    if args.train_data_path is not None:
        dataset_train = MRIDataset(config, args, training=True)
        loader_train = DataLoader(dataset_train,
                                  batch_size=config.batch_size,
                                  sampler=RandomSampler(dataset_train),
                                  collate_fn=dataset_train.collate_fn,
                                  pin_memory=config.pin_mem,
                                  num_workers=config.num_cpu_workers)

    if args.val_data_path is not None:
        dataset_val = MRIDataset(config, args, validation=True)
        loader_val = DataLoader(dataset_val,
                                batch_size=config.batch_size,
                                collate_fn=dataset_val.collate_fn,
                                pin_memory=config.pin_mem,
                                num_workers=config.num_cpu_workers)

    if args.test_data_path is not None:
        dataset_test = MRIDataset(config, args, test=True)
        loader_test = DataLoader(dataset_test,
                                batch_size=config.batch_size,
                                collate_fn=dataset_test.collate_fn,
                                pin_memory=config.pin_mem,
                                num_workers=config.num_cpu_workers)

    model = DLModel(net, loss, config, args,
                         loader_train=loader_train,
                         loader_val=loader_val,
                         loader_test=loader_test,
                         scheduler=scheduler)

    if args.train:
        model.training()
        path_to_save = os.path.join(args.checkpoint_dir, args.exp_name, 'model.ptc')
        model.save_model(path_to_save)
    if args.test:
        model.testing()