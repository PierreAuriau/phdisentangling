#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:18:09 2022


@source : https://github.com/Duplums/bhb10k-dl-benchmark/blob/main/dl_model.py
"""

import os
import torch
from torch.nn import DataParallel
from metrics import METRICS
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from early_stopping import EarlyStopping


class DLModel:

    def __init__(self, net, loss, config, args, loader_train=None, loader_val=None, loader_test=None, scheduler=None, log_dir=''):
        """
        Parameters
        ----------
        net: subclass of nn.Module
        loss: callable fn with args (y_pred, y_true)
        config: Config object with hyperparameters
        args: args given including data pathss
        loader_train, loader_val, loader_test: pytorch DataLoaders for training/validation/test
        scheduler (optional)
        """
        super().__init__()
        self.logger = logging.getLogger("DLModel")
        self.logger.setLevel(logging.INFO)
        self.loss = loss
        self.model = net
        self.optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        ##Old :
        # self.scheduler = scheduler
        if scheduler is not None :
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=config.gamma_scheduler, step_size=config.step_size_scheduler)
        else:
            self.scheduler = scheduler
        self.loader = loader_train
        self.loader_val = loader_val
        self.loader_test = loader_test
        self.model_path = args.model_path
        self.device = torch.device("cuda" if config.cuda else "cpu")
        if config.cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: set cuda=False parameter.")
        print('Device : ', self.device)
        self.config = config
        self.metrics = {m: METRICS[m] for m in args.metrics}
        self.model = DataParallel(self.model).to(self.device)
        
        self.log_dir = log_dir
        self.test_data_path = args.test_data_path if args.test else None

    def training(self):
        print('Loss : ', self.loss)
        print('Optimizer :', self.optimizer)
        print('Scheduler :', self.scheduler)
        
        #tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        #early stopping
        if self.config.early_stopping:
            early_stopping = EarlyStopping(patience=self.config.es_patience, verbose=True)

        for epoch in range(self.config.nb_epochs):
            ## Training step
            self.model.train()
            nb_batch = len(self.loader)
            #training_loss = {}
            training_loss = 0
            pbar = tqdm(total=nb_batch, desc="Training")
            for (inputs, labels) in self.loader:
                pbar.update()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                y = self.model(inputs)
                batch_loss = self.loss(y, labels)
                batch_loss.backward()
                self.optimizer.step()
                training_loss += float(batch_loss) / nb_batch
                #training_loss += [float(batch_loss) / nb_batch]
            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation")
            val_loss = 0
            y_val = []
            y_true_val = []
            with torch.no_grad():
                self.model.eval()
                for (inputs, labels) in self.loader_val:
                    pbar.update()
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    y = self.model(inputs)
                    y_val.extend(y)
                    y_true_val.extend(labels)
                    batch_loss = self.loss(y, labels)
                    val_loss += float(batch_loss) / nb_batch
            pbar.close()
            all_metrics = dict()
            for name, metric in self.metrics.items():
                all_metrics[name] = metric(torch.tensor(y_val), torch.tensor(y_true_val))
            all_metrics_str = "\t".join(["{}={:.2f}".format(name, m) for (name, m)in all_metrics.items()])
            print("Epoch [{}/{}] Training loss = {:.4f}\t Validation loss = {:.4f}\t".format(
                epoch+1, self.config.nb_epochs, training_loss, val_loss)+"\t".join(all_metrics)+all_metrics_str,
                  flush=True)
            
            #tensorboard
            self.writer.add_scalar('loss/training', training_loss, epoch)
            self.writer.add_scalar('loss/validation', val_loss, epoch)
            for name, metric in all_metrics.items():
                self.writer.add_scalar(name + '/validation', metric, epoch)
                
            #early stopping
            if self.config.early_stopping:
                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:   
                    break
            
            #scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        #tensorboard
        self.writer.close()


    def testing(self, name=''):
        self.load_model(self.model_path)
        nb_batch = len(self.loader_test)
        pbar = tqdm(total=nb_batch, desc="Test")
        y_pred = []
        y_true = []
        with torch.no_grad():
            self.model.eval()
            # /!\ old: for (inputs, labels) in self.loader_val:
            for (inputs, labels) in self.loader_test:
                pbar.update()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                y = self.model(inputs)
                y_pred.extend(y)
                y_true.extend(labels)
        pbar.close()
        all_metrics = dict()
        for name, metric in self.metrics.items():
            all_metrics[name] = metric(torch.tensor(y_pred), torch.tensor(y_true))
        all_metrics_str = "\t".join(["{}={:.2f}".format(name, m) for (name, m) in all_metrics.items()])
        print(all_metrics_str, flush=True)
        
        #saving results
        path_to_results = os.path.join(self.log_dir, "test" + name + ".csv")
        df_results = pd.DataFrame({'dataset': [self.test_data_path]})
        for name,metric in all_metrics.items():
            df_results[name] = metric
        if os.path.exists(path_to_results):
            df_previous_results = pd.read_csv(path_to_results)
            df_results = pd.concat([df_previous_results, df_results], ignore_index=True)
        df_results.to_csv(path_to_results, index=False)
        

    def load_model(self, path):
        self.load_model(torch.load(path))
        self.logger.info('Model loaded : {}'.format(path))
        # checkpoint = None
        # try:
        #     checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        # except BaseException as e:
        #     self.logger.error('Impossible to load the checkpoint: %s' % str(e))
        # if checkpoint is not None:
        #     try:
        #         if hasattr(checkpoint, "state_dict"):
        #             unexpected = self.model.load_state_dict(checkpoint.state_dict())
        #             self.logger.info('Model loading info: {}'.format(unexpected))
        #         elif isinstance(checkpoint, dict):
        #             if "model" in checkpoint:
        #                 unexpected = self.model.load_state_dict(checkpoint["model"], strict=False)
        #                 self.logger.info('Model loading info: {}'.format(unexpected))
        #         else:
        #             unexpected = self.model.load_state_dict(checkpoint)
        #             self.logger.info('Model loading info: {}'.format(unexpected))
        #     except BaseException as e:
        #         raise ValueError('Error while loading the model\'s weights: %s' % str(e))
                
    def save_model(self, path):
      torch.save(self.model.state_dict(), path)
      self.logger.info('Model saved')