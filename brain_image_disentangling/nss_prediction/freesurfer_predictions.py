"""
We aim at giving learning curves of classical ML models on AUSZ for:
* Prediction of NSS
* VBM pre-processing

"""
import os
import sys
import re
import argparse
import pandas as pd
import numpy as np

import nibabel
from collections import defaultdict
import itertools
from copy import deepcopy
from multiprocessing import Pool

import sklearn.linear_model as lm
from sklearn import preprocessing
import sklearn.svm as svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, balanced_accuracy_score

from config import Config

config = Config()

def load_data():
    # Loading data
    try:
        arr = np.load(os.path.join(config.tmp, "freesurfer", "ausz_freesurfer_thickness_curv_area_sulc.npy"), mmap_mode="r")
        with open(os.path.join(config.tmp, "freesurfer", "channels.txt"), "r") as f:
            channels = [re.search("texture-([a-z]+)", l).group(1) for l in f.readlines()]
        df = pd.read_csv(os.path.join(config.tmp, "freesurfer", "ausz_freesurfer_participants.csv"), dtype=config.id_types)
        scheme = pd.read_csv(os.path.join(config.tmp, "processed", "nss_stratified_10_fold_ausz.csv"), dtype=config.id_types)
        print(f"WARNING: using tmp directory : {config.tmp}")    
    except FileNotFoundError:
        raise FileNotFoundError(f"Data not found in : {config.root}")
    print(f"Data shape : {arr.shape} | {len(df)} | {len(scheme)}")
    print(f"Channels: {channels}")
    assert (df[["participant_id", "session"]] == scheme[["participant_id", "session"]]).all().all(), \
           print("Scheme and participant dataframe do not have same order.")
    return arr, channels, df, scheme

def get_model_cv(model, cv_train=3):
    models_cv =  dict(
    logreg_cv=make_pipeline(
        preprocessing.StandardScaler(),
        GridSearchCV(lm.LogisticRegression(max_iter=1000),
                     param_grid={"C": 10. ** np.arange(-1, 3)},
                     cv=cv_train, n_jobs=1)),
    lrl2_cv=make_pipeline(
        preprocessing.StandardScaler(),
        # preprocessing.MinMaxScaler(),
        GridSearchCV(lm.Ridge(),
                     param_grid={'alpha': 10. ** np.arange(-1, 3)},
                     cv=cv_train, n_jobs=1)),

    lrenet_cv=make_pipeline(
        preprocessing.StandardScaler(),
        # preprocessing.MinMaxScaler(),
        GridSearchCV(lm.ElasticNet(max_iter=1000),
                     param_grid={'alpha': 10. ** np.arange(-1, 2),
                                 'l1_ratio': [.1, .5]},
                     cv=cv_train, n_jobs=1)),

    svmrbf_cv=make_pipeline(
        # preprocessing.StandardScaler(),
        preprocessing.MinMaxScaler(),
        GridSearchCV(svm.SVR(),
                     # {'kernel': ['poly', 'rbf'], 'C': 10. ** np.arange(-3, 3)},
                     param_grid={'kernel': ['poly', 'rbf'],
                                 'C': 10. ** np.arange(-1, 2)},
                     cv=cv_train, n_jobs=1)),

    forest_cv=make_pipeline(
        # preprocessing.StandardScaler(),
        preprocessing.MinMaxScaler(),
        GridSearchCV(RandomForestRegressor(random_state=1),
                     param_grid={"n_estimators": [100]},
                     cv=cv_train, n_jobs=1)),

    gb_cv=make_pipeline(
        preprocessing.MinMaxScaler(),
        GridSearchCV(estimator=GradientBoostingRegressor(random_state=1),
                     param_grid={"n_estimators": [100],
                                 "subsample":[1, .5],
                                 "learning_rate": [.1, .5]
                                 },
                     cv=cv_train, n_jobs=1)))
    return models_cv.get(model+"_cv")

def train(arr, channels, df, scheme, model, label, texture, saving_dir):

    mapping = {
       "M": 0,
       "m": 0,
       "F": 1,
       "control": 0,
       "scz": 1,
       "scz-asd": 1,
       "asd": 1
    }

    model_cv = get_model_cv(model)
    i_texture = channels.index(texture)
    
    nb_folds = 10
    logs = defaultdict(list)
    
    for fold in range(nb_folds):
        print(f"# Fold: {fold}")

        # 0) Load data
        train_mask = scheme[f"fold{fold}"] == "train"
        train_data = arr[train_mask, i_texture, ...]
        y_train = df.loc[train_mask, label]

        test_mask = scheme[f"fold{fold}"] == "test"
        test_data = arr[test_mask, i_texture, ...]
        y_test = df.loc[test_mask, label]

        print(train_data.shape, test_data.shape)
        
        if label in ["sex", "diagnosis"]:
            y_train = y_train.replace(mapping).values.astype(int)
            y_test = y_test.replace(mapping).values.astype(int)
            
            metrics = {
                "roc_auc": lambda y_pred, y_true: roc_auc_score(y_true=y_true, y_score=y_pred[:, 1]),
                "bacc": lambda y_pred, y_true: balanced_accuracy_score(y_true=y_true, y_pred=(y_pred[:, 1] > 0.5).astype(int))
            }
            
        else:
            y_train = y_train.values.astype(np.float32)
            y_test = y_test.values.astype(np.float32)
            
            metrics = {
                "r2": r2_score,
                "rmse": lambda y_pred, y_true: mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False),
                "mae": mean_absolute_error
            }

        # 2) Training    
        model_cv.fit(train_data, y_train)
        # best_score, best_params = model_cv[1].best_score_, model_cv[1].best_params_
        # print(f"Model trained: best score : {best_score:.2g} - best params : {best_params}")

        # 3) Testing
        print(f"Score on test set: {model_cv.score(test_data, y_test):.2f}")
        for split, data, y_true in zip(["train", "test"], 
                                       [train_data, test_data], 
                                       [y_train, y_test]):
            if label in ["sex", "diagnosis"]:
                y_pred = model_cv.predict_proba(data)
            else:
                y_pred = model_cv.predict(data)

            for name, metric in metrics.items():
                try:
                    value = metric(y_true=y_true, y_pred=y_pred)
                except ValueError as e:
                    print(f"ValueError during evaluation: {e}")
                    continue

                # 4) Saving
                logs["label"].append(label)
                logs["texture"].append(texture)
                logs["fold"].append(fold)
                logs["split"].append(split)
                logs["metric"].append(name)
                logs["value"].append(value)
    
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(os.path.join(saving_dir, f"texture-{texture}_model-{model}_label-{label}.csv"), 
                                sep=",", index=False)



def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--saving_dir", required=True, type=str, help="Saving directory.")
    parser.add_argument("--models", type=str, nargs="+", default=["lrl2"], help="Models to trained", 
                         choices=["lrl2", "lrenet", "svmrbf", "forest", "gb"])
    parser.add_argument("--labels", type=str, nargs="+", default=["NSS"], help="Labels to predict", 
                         choices=["NSS", "age", "sex", "diagnosis", "tiv", "skeleton_size"])
    parser.add_argument("--n_permutations", type=int, default=0)
    parser.add_argument("--textures", type=str, nargs="+", help="Freesurfer texture inputs", 
                         choices=["curv", "area", "thickness", "sulc"])
    args = parser.parse_args(argv)

    return args

if __name__ == "__main__":

    args = parse_args(sys.argv[1:])
       
    saving_dir = os.path.join(config.path2models, "freesurfer", args.saving_dir)
    labels = args.labels
    n_permutations = args.n_permutations + 1
    textures = args.textures
    
    os.makedirs(saving_dir, exist_ok=True)
    print("Data Loading")
    arr, channels, df, scheme = load_data()
    
    if n_permutations > 1:
        label = "NSS"
        permutations = {}
        permutations["permutation-0"] = df[label].values
        for i in range(1, n_permutations):
            permutations[f"permutation-{i}"] = np.random.permutation(df[label].values)
        permutations_df = pd.DataFrame(permutations)
        df = pd.concat((df, permutations_df), axis=1)
        labels += [f"permutation-{i}" for i in range(n_permutations)]
    models = ["logreg" if l in ["diagnosis", "sex"] else "lrl2" for l in labels]
    
    print("Models training")
    # train(arr, channels, df, scheme, 'lrl2', 'NSS', 'sulc', saving_dir)
    
    list_args = [(arr, channels, df, scheme, mdl, lbl, txt, saving_dir) 
                 for txt, (mdl, lbl) in itertools.product(textures, zip(models, labels))]

    with Pool() as pool:
        pool.starmap(train, list_args)
    