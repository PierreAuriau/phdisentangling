"""
We aim at giving learning curves of classical ML models on AUSZ for:
* Prediction of sex, age, tiv, skeleton_size
* Skeleton pre-processing

"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

import nibabel as nib
from collections import defaultdict
from multiprocessing import Pool

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.ndimage import binary_dilation, gaussian_filter

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, balanced_accuracy_score

from config import Config

config = Config()

def load_data():
    # Loading data
    try:
        arr = np.load(os.path.join(config.root, "ausz_t1mri_skeleton_data32.npy"), mmap_mode="r")
        df = pd.read_csv(os.path.join(config.root, "ausz_t1mri_participants.csv"), dtype=config.id_types)
        scheme = pd.read_csv(os.path.join(config.root, "stratified_10_fold_ausz.csv"), dtype=config.id_types)    
        m_vbm = nib.load(os.path.join(config.root, "mni_cerebrum-gm-mask_1.5mm.nii.gz"))
        brain_mask = (m_vbm.get_fdata() != 0).astype(bool)
        brain_mask_pad = np.pad(brain_mask, pad_width = ((3, 4), (3, 4), (3, 4)))
        print(f"Brain mask shape : {brain_mask.shape}, with padding : {brain_mask_pad.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Data not found in : {config.root}")
    print(f"Data shape : {arr.shape} | {len(df)} | {len(scheme)}") # FIXME
    arr = arr[..., :, 4:156, :]
    smoothed_arr = np.zeros_like(arr)
    for i, img in enumerate(arr):  
        smoothed_arr[i, 0, ...] = gaussian_filter(img[0], sigma=(1.0, 1.0, 1.0), mode="constant", cval=0.0, radius=(2, 2, 2))
    # final brain mask
    final_brain_mask = binary_dilation(brain_mask_pad, iterations=4, border_value=0, origin=0)
    # reshaping data with final brain mask
    arr = arr[:, 0, final_brain_mask]
    arr = arr.astype(np.float32)
    smoothed_arr = smoothed_arr[:, 0, final_brain_mask]
    smoothed_arr = smoothed_arr.astype(np.float32)
    print(f"Data shape after applying mask: {arr.shape} | {len(df)} | {len(scheme)}")
    assert (df[["participant_id", "session"]] == scheme[["participant_id", "session"]]).all().all(), \
           print("Scheme and participant dataframe do not have same order.")
    return arr, smoothed_arr, df, scheme

def train(arr, df, scheme, label, scaler, saving_dir):
    mapping = {
    "M": 0,
    "m": 0,
    "F": 1,
    "control": 0,
    "scz": 1,
    "scz-asd": 1,
    "asd": 1
    }
    nb_folds = 10
    logs = defaultdict(list)
    for fold in range(nb_folds):
        print(f"# Fold: {fold}")

        # 0) Load data
        train_mask = scheme[f"fold{fold}"] == "train"
        train_data = arr[train_mask]
        y_train = df.loc[train_mask, label]

        test_mask = scheme[f"fold{fold}"] == "test"
        test_data = arr[test_mask]
        y_test = df.loc[test_mask, label]
        
        if label in ["sex", "diagnosis"]:
            y_train = y_train.replace(mapping).values.astype(int)
            y_test = y_test.replace(mapping).values.astype(int)

            model = LogisticRegression(C=1.0, max_iter=1000, solver="sag", penalty="l2", fit_intercept=True, class_weight="balanced")

            metrics = {
                "roc_auc": lambda y_pred, y_true: roc_auc_score(y_true=y_true, y_score=y_pred[:, 1]),
                "bacc": lambda y_pred, y_true: balanced_accuracy_score(y_true=y_true, y_pred=(y_pred[:, 1] > 0.5).astype(int))
            }
        else:
            y_train = y_train.values.astype(np.float32)
            y_test = y_test.values.astype(np.float32)

            model = GridSearchCV(Ridge(),
                                 param_grid={'alpha': 10. ** np.arange(-1, 3)},
                                 cv=3, n_jobs=3)
            
            metrics = {
                "r2": r2_score,
                "rmse": lambda y_pred, y_true: mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False),
                "mae": mean_absolute_error
            }
        
        # 1) Normalization
        if scaler:
            scaler = StandardScaler
            ss = scaler().fit(train_data)
            train_data = ss.transform(train_data)
            test_data = ss.transform(test_data)

        # 2) Training    
        model.fit(train_data, y_train)

        # 3) Testing
        print(f"Score on test set: {model.score(test_data, y_test)}")
        for split, data, y_true in zip(["train", "test"], 
                                       [train_data, test_data], 
                                       [y_train, y_test]):
            if label in ["sex", "diagnosis"]:
                y_pred = model.predict_proba(data)
            else:
                y_pred = model.predict(data)

            for name, metric in metrics.items():
                try:
                    value = metric(y_true=y_true, y_pred=y_pred)
                except ValueError as e:
                    print(f"ValueError during evaluation: {e}")
                    continue

                # 4) Saving
                logs["label"].append(label)
                logs["fold"].append(fold)
                logs["split"].append(split)
                logs["metric"].append(name)
                logs["value"].append(value)

    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(os.path.join(saving_dir, f"preproc-skeleton_model-lrl2_label-{label}.csv"), 
                                sep=",", index=False)

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--saving_dir", required=True, type=str, help="Saving directory.")
    parser.add_argument("--labels", type=str, nargs="+", default=["NSS"], help="Labels to predict", 
                         choices=["NSS", "age", "sex", "diagnosis", "tiv", "skeleton_size"])
    parser.add_argument("--scaler", action="store_true", help="Scale the data to train the model.")
    parser.add_argument("--n_permutations", type=int, default=1000)
    args = parser.parse_args(argv)

    return args

if __name__ == "__main__":

    args = parse_args(sys.argv[1:])
    saving_dir = os.path.join(config.path2models, "skeleton", args.saving_dir)
    label = args.labels[0]
    n_permutations = args.n_permutations + 1

    os.makedirs(saving_dir, exist_ok=True)
    print("Data Loading")
    arr, smoothed_arr, df, scheme = load_data()
    
    permutations = {}
    permutations["permutation-0"] = df[label].values
    for i in range(1, n_permutations):
        permutations[f"permutation-{i}"] = np.random.permutation(df[label].values)
    permutations_df = pd.DataFrame(permutations)
    df = pd.concat((df, permutations_df), axis=1)
    
    print("Models training")
    list_args = [(smoothed_arr, df, scheme, f"permutation-{i}", args.scaler, saving_dir) for i in range(n_permutations)]

    with Pool() as pool:
        pool.starmap(train, list_args)
