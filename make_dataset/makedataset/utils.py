#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to make datasets
"""
from collections import OrderedDict
import re
import numpy as np
import pandas as pd

def get_keys(filename, id_regex, ses_regex, acq_regex, run_regex):
    """
    Extract keys from bids filename. Check consistency of filename.
    """

    keys = OrderedDict()

    participant_id = re.compile(id_regex).findall(filename)
    if len(set(participant_id)) != 1:
        raise ValueError('Found several or no participant id', participant_id, 'in path', filename)
    keys["participant_id"] = participant_id[0]

    session = re.compile(ses_regex).findall(filename)
    if len(set(session)) > 1:
        raise ValueError('Found several sessions', session, 'in path', filename)

    elif len(set(session)) == 1:
        keys["session"] = session[0]

    else:
        keys["session"] = ""

    acquisition = re.compile(acq_regex).findall(filename)
    if len(set(acquisition)) == 1:
        keys["acq"] = acquisition[0]
    else:
        keys["acq"] = ""

    run = re.compile(run_regex).findall(filename)
    if len(set(run)) == 1:
        keys["run"] = run[0]

    else:
        keys["run"] = ""

    keys["ni_path"] = filename

    return keys

def is_it_a_subject(filename):
    if re.search('.minf$', filename):
        return False
    elif re.search('.sqlite$', filename):
        return False
    elif re.search('.html$', filename):
        return False
    elif re.search("snapshots", filename):
        return False
    else:
        return True

def diff_sets(a, b):
    """Compare sets

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.

    Returns
    -------
    set
        diff between a and b.
    set
        a - b.
    set
        b - a.
    """
    a = set(a)
    b = set(b)
    return (a - b).union(b - a), a - b, b - a

def global_scaling(NI_arr, axis0_values=None, target=1500):
    """
    Apply a global proportional scaling, such that axis0_values * gscaling == target
    Parameters
    ----------
    NI_arr:  ndarray, of shape (n_subjects, 1, image_shape).
    axis0_values: 1-d array, if None (default) use global average per subject: NI_arr.mean(axis=1)
    target: scalar, the desired target
    Returns
    -------
    The scaled array
    >>> import numpy as np
    >>> import brainomics.image_preprocessing as preproc
    >>> NI_arr = np.array([[9., 11], [0, 2],  [4, 6]])
    >>> NI_arr
    array([[ 9., 11.],
           [ 0.,  2.],
           [ 4.,  6.]])
    >>> axis0_values = [10, 1, 5]
    >>> preproc.global_scaling(NI_arr, axis0_values, target=1)
    array([[0.9, 1.1],
           [0. , 2. ],
           [0.8, 1.2]])
    >>> preproc.global_scaling(NI_arr, axis0_values=None, target=1)
    array([[0.9, 1.1],
           [0. , 2. ],
           [0.8, 1.2]])
    """
    if axis0_values is None:
        axis0_values = NI_arr.mean(axis=1)
    gscaling = target / np.asarray(axis0_values)
    gscaling = gscaling.reshape([gscaling.shape[0]] + [1] * (NI_arr.ndim - 1))
    return gscaling * NI_arr

def ml_regression(data, y):
    """ Basic QC for age predictio

    data : dict of arrays (N, P)
    y : array (N, )
    """
    # sklearn for QC
    import sklearn.linear_model as lm
    from sklearn.model_selection import cross_validate
    from sklearn import preprocessing
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import KFold

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    lr = make_pipeline(preprocessing.StandardScaler(), lm.Ridge(alpha=10))

    res = list() #pd.DataFrame(columns= ["r2", "mae", "rmse"])
    for name, X, in sorted(data.items()):
        cv_res = cross_validate(estimator=lr, X=X, y=y, cv=cv,
                                n_jobs=5,
                                scoring=['r2', 'neg_mean_absolute_error',
                                         'neg_mean_squared_error'])
        r2 = cv_res['test_r2'].mean()
        rmse = np.sqrt(np.mean(-cv_res['test_neg_mean_squared_error']))
        mae = np.mean(-cv_res['test_neg_mean_absolute_error'])
        res.append([name, r2, mae, rmse])
        print("%s:\tCV R2:%.4f, MAE:%.4f, RMSE:%.4f" % (name, r2, mae, rmse))

    return pd.DataFrame(res, columns= ["data", "r2", "mae", "rmse"])


def ml_correlation_plot(data, y, output, study):
    # to understand why bas R2
    # sklearn for QC
    import sklearn.linear_model as lm
    from sklearn.model_selection import cross_val_predict
    from sklearn import preprocessing
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt
    matplotlib.use( 'tkagg' )

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    lr = make_pipeline(preprocessing.StandardScaler(), lm.Ridge(alpha=10))

    for name, X, in data.items():
        predicted = cross_val_predict(lr, X, y, cv=cv)
        fig, ax = plt.subplots()
        ax.scatter(y, predicted, edgecolors=(0, 0, 0))
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        ax.set_xlabel('Measured (Age)')
        ax.set_ylabel('Predicted (Brain Age)')
        plt.title(name)
        plt.savefig(os.path.join(output, "{0}_{1}_corr_plot".format(study, name)))
        # plt.show()

def df_column_switch(df, column1, column2):
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df