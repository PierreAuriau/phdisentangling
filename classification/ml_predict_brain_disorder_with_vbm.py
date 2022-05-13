#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pierre
"""

#Module import

import os
import os.path as op
# import tempfile
#import urllib.request
#import glob
#import time
#from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

#Neuroimaging
import nibabel
#from nitk.image import img_to_array, global_scaling, compute_brain_mask, rm_small_clusters
#from nitk.bids import get_keys
#from nitk.data import fetch_data

# Models
#from sklearn.decomposition import PCA
import sklearn.linear_model as lm
#import sklearn.svm as svm
#from sklearn.neural_network import MLPClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier

# Univariate statistics
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#import statsmodels.stats.api as sms

# Metrics
import sklearn.metrics as metrics

# Resampling
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import cross_validate
#from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline


###################
### Data import ###
###################
if op.exists('/neurospin'):
    prefixe = ''
else:
    prefixe = '/home/pa267054'
    
studies = {"biobd", "bsnip1", "schizconnect-vip-prague"}

path = op.join(prefixe, '/neurospin/dico/pauriau/data')
participants_filename = '_cat12vbm_participants.csv'
rois_filename = '_cat12vbm_rois-gs.csv'
mwp1_filename = '_cat12vbm_mwp1-gs.npy'
mask_filename = op.join(prefixe, '/neurospin/psy/ni_ressources/masks/mni_brain-mask_1.5mm.nii.gz')


######################
### Classification ###
######################

def cv_train_test_scores_params_classif(model, y_train, y_pred_train,
                                        score_pred_train, y_test, y_pred_test,
                                        score_pred_test):
    """Compute CV score, train and test score from a cv grid search model.

    Parameters
    ----------
    model : Pipeline or GridSearchCV
        Model. If Pipeline, assume the predictive model is the last step.
    y_train : array
        True train values.
    y_pred_train : array
        Predicted train values.
    score_pred_train : array
        predicted decision function on train set.
    y_test : array
        True test values.
    y_pred_test : array
        Predicted test values.
    score_pred_test : array
        predicted decision function on test set.

    Returns
    -------
    info : DataFrame
        DataFrame(acc_cv, acc_train, bacc_train, auc_train, acc_test,
                  bacc_test, auc_test, best_params).
    """
    # fetch predictor (second items in the pipeline)
    # If the model is a pipeline pick the last step, ie the predictor
    gridsearchcv = model.steps[-1][1] if hasattr(model, "steps") else model
    # Default estimator’s score method is used: accuracy
    acc_cv = gridsearchcv.best_score_
    # and best params
    best_params = gridsearchcv.best_params_

    # Train scores
    acc_train = metrics.accuracy_score(y_train, y_pred_train)
    bacc_train = metrics.balanced_accuracy_score(y_train, y_pred_train)
    auc_train = metrics.roc_auc_score(y_train, score_pred_train)

    # Test scores
    acc_test = metrics.accuracy_score(y_test, y_pred_test)
    bacc_test = metrics.balanced_accuracy_score(y_test, y_pred_test)
    auc_test = metrics.roc_auc_score(y_test, score_pred_test)

    info = pd.DataFrame([[acc_cv, acc_train, bacc_train, auc_train,
                          acc_test, bacc_test, auc_test,
                          str(best_params)]],
                        columns=("acc_cv", "acc_train", "bacc_train", "auc_train",
                                 "acc_test", "bacc_test", "auc_test",
                                 "best_params"))
    return info


for s in studies:
    print('===== STUDY: ', s, ' =====')
    participants = pd.read_csv(op.join(path, s, s+participants_filename))
    rois = pd.read_csv(op.join(path, s, s+rois_filename))
    imgs_arr = np.load(op.join(path, s, s+mwp1_filename))
    mask_img = nibabel.load(mask_filename)
    mask_arr = mask_img.get_fdata() != 0

    diagnosis = participants.loc[:, 'diagnosis'].values
    diagnosis = np.array(diagnosis ==  'control', dtype=int)

    # Linear Regression
    
    N_FOLDS = 5
    cv_train = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    
    
    data = {'rois': rois.loc[:, 'l3thVen_GM_Vol':].values,
            'vbm': imgs_arr.squeeze()[:, mask_arr]}
    
    for name, X in data.items():
        X_train, X_test, y_train, y_test = train_test_split(X, diagnosis, test_size=0.2, train_size=0.8, random_state=None, shuffle=True)
    
        lrl2_cv = make_pipeline(
            preprocessing.StandardScaler(),
            GridSearchCV(lm.LogisticRegression(max_iter=10000),
                         {'C': 10. ** np.arange(-3, 3)},
                         cv=cv_train, n_jobs=N_FOLDS))
    
    
        lrl2_cv.fit(X=X_train, y=y_train)
    
        y_pred_train = lrl2_cv.predict(X_train)
        y_pred_test = lrl2_cv.predict(X_test)
        score_pred_train = lrl2_cv.decision_function(X_train)
        score_pred_test = lrl2_cv.decision_function(X_test)
    
        print('### Data: ', name, ' ###')
        info = cv_train_test_scores_params_classif(lrl2_cv, y_train, y_pred_train,
                                            score_pred_train, y_test, y_pred_test,
                                            score_pred_test)
    
        print(info[['acc_cv', 'best_params']])
        print('* Train: *')
        print(info[['acc_train', 'bacc_train', 'auc_train']])
        print('* Test: *')
        print(info[['acc_test', 'bacc_test', 'auc_test']])
        
        print('### Save results ###')
        info.to_csv(s + '_' + name + '_results.csv')
