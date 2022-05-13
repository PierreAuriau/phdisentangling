#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Module import

import os
import os.path as op
import sys
# import tempfile
#import urllib.request
import glob
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

#NS_dataset
from parse_study import parse_study

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


from tools import quasi_raw_nii2npy

###################
### Data import ###
###################
if op.exists('/neurospin'):
    prefixe = ''
else:
    prefixe = '/home/pa267054'
    
studies = {"biobd", "bsnip1", "schizconnect-vip-prague"}

study = "biobd"
path = op.join(prefixe, "neurospin/psy_sbox", study, "derivatives/quasi-raw/subjects")


path_to_file = "/home/pa267054/neurospin/psy_sbox/bsnip1/derivatives/quasi-raw/sub-INV0AL14J6U/ses-1/anat/sub-INV0AL14J6U_ses-1_acq-103_run-1_desc-6apply_T1w.nii.gz"
dataset_dir = op.join(prefixe, "neurospin/psy_sbox/datasets")
im = nibabel.load(path_to_file)
view_data = im.get_fdata()
data = np.asarray(view_data)

mask_im = nibabel.load(os.path.join(dataset_dir, "MNI152_T1_1mm_brain_mask.nii.gz"))
mask_arr = (mask_im.get_fdata() != 0)
img = data.squeeze()[mask_arr]

#################



#### Make numpy file ####

study_dir = op.join(prefixe, "neurospin/psy_sbox", study)

regex = "derivatives/quasi-raw/sub-12*/ses*/anat/*preproc-linear*.nii.gz"

qc_file = "derivatives/cat12-12.6_vbm_qc/qc.tsv"

nii_path = op.join(study_dir, regex)
phenotype_filename = op.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')
output_path = op.join(prefixe, 'neurospin/dico/pauriau/data', study)
dataset_name = study
qc = op.join(study_dir, qc_file)
quasi_raw_nii2npy(nii_path, phenotype, dataset_name, output_path, qc=qc, sep='\t', id_type=str,
                 check = dict(shape=(182, 218, 182), zooms=(1, 1, 1)))


#### Classification ###

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



# Import numpy file
quasiraw = np.load(op.join(output_path, 'biobd_cat12vbm_quasi_raw_data64.npy'))
# Apply mask
quasiraw_img = quasiraw.squeeze()[:, mask_arr]

# Import participants file
#to modify
participants = pd.read_csv(op.join(output_path, 'biobd_cat12vbm_quasi_raw_participants.tsv'), sep='\t')

diagnosis = participants.loc[:, 'diagnosis'].values
diagnosis = np.array(diagnosis ==  'control', dtype=int)

# Linear Regression

N_FOLDS = 5
cv_train = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(quasiraw_img, diagnosis, test_size=0.2, train_size=0.8, random_state=None, shuffle=True)

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

info = cv_train_test_scores_params_classif(lrl2_cv, y_train, y_pred_train,
                                    score_pred_train, y_test, y_pred_test,
                                    score_pred_test)

print(info[['acc_cv', 'best_params']])
print('* Train: *')
print(info[['acc_train', 'bacc_train', 'auc_train']])
print('* Test: *')
print(info[['acc_test', 'bacc_test', 'auc_test']])

print('### Save results ###')
info.to_csv(study + '_results.csv')






