#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 11:20:32 2022

@source: https://github.com/brainvisa/morpho-deepsulci/blob/master/python/deepsulci/deeptools/early_stopping.py
"""

import numpy as np
import torch
import logging

class EarlyStopping(object):
    """
    Early stops the training if validation loss doesn't improve after
    a given patience.
    """
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation
                            loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation
                            loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.logger = logging.getLogger("EarlyStopping")

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                self.logger.info('Counter: %i out of %i' %
                      (self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.logger.info('Stopping the training')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            #self.logger.info('Validation loss decreased (%.6f -> %.6f). Saving model...' %
            #      (self.val_loss_min, val_loss))
            pass
        #torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss