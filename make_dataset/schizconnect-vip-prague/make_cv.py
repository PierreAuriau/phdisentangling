#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:32:06 2022

@author: pa267054
"""

import os.path as op
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


N_FOLDS = 5
X_path = '/neurospin/dico/pauriau/data/cohorts/schizconnect-vip-prague/schizconnect-vip-prague_cat12vbm_mwp1-gs.npy'
y_path = '/neurospin/dico/pauriau/data/cohorts/schizconnect-vip-prague/schizconnect-vip-prague_cat12vbm_participants.csv'
path_to_save = '/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/schizconnect-vip-prague'

X = np.load(X_path)
y = pd.read_csv(y_path)

y['diagnosis'] = np.array(y['diagnosis'] ==  'control', dtype=int)

y_arr = y['diagnosis']

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for i, (train_index, val_index) in enumerate(skf.split(X, y_arr)):
    X_train, X_val = X[train_index], X[val_index] 
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    np.save(op.join(path_to_save, 'X_train_' + str(i) + '.npy' ), X_train)
    np.save(op.join(path_to_save, 'X_val_' + str(i) + '.npy' ), X_val)
    y_train.to_csv(op.join(path_to_save, 'y_train_' + str(i) + '.csv'))
    y_val.to_csv(op.join(path_to_save, 'y_val_' + str(i) + '.csv'))
