# Predictions of neurological soft sign scores

We aimed to predict the neurological soft sign (NSS) scores of subjects from brain anatomical features. 
We compared the prediction scores from several brain features, namely skeletons, VBM, Freesurfer (cortical thickness, area, curvature).

## Dataset

The dataset is part of the study "From Autism to Schizophrenia" (AUSZ) from the Sainte-Anne's Hospital, Paris, France.
Four groups were defined:
* ASD: patients with autism spectrum disorders without intellectual deficit
* SCZ-ASD: patients with adult-onset schizophrenia after an early-adolescence prodromal phase that began before age 15
* SCZ: patients with adult-onset schizophrenia after an late-adolescence prodromal phase that began after age 15
* HC: healthy controls


| Diagnosis | N | Age | Sex (%F) | NSS scores |
| :---: | :---: | :---: | :---: | :---: |
| Control | 26 | 23.2 ±2.9 | 19.2 | 5.1 ±3.7 |
| ASD | 25 | 22.0 ±3.0 | 12.0 | 17.2 ±6.3 |
| SCZ-ASD | 20 | 22.7 ±3.3 | 20.0 | 11.9 ±8.5 |
| SCZ | 24 | 23.7 ±3.7 | 12.5 | 10.8 ±6.0 |
| Total | 95 | 22.9 ±3.3 | 15.8 | 11.2 ±7.5 |
  
## Methods

We implemeted a linear regressions from the brain anatomical features with a l2 regularization ([Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) model from scikit-learn) to predict the NSS scores.
For the skeletons, we also implemented a deep learning (DL) model with a Densenet-121 architecture, L1 loss and a l2 regularization. We tried random initialization of the weights and initialization with a self-supervised pretraining.

## Results

| Brain feature | Method | Mean absolute error (MAE) |
| :--- | :---: | :---: |
| VBM | linear regression | 5.86 ±1.01 |
| Skeleton | linear regression | 5,74 ±0,92 |
|  | DL w/o pretraining | 5.90 ±1.44 |
|  | DL with pretraining | 5.75 ±0.84 |
| Freesurfer: | | |
| - curvature | linear regression | 5.98 ±0.89 | 
| - cortical thickness | linear regression | 5.69 ±0.93 |
| - area | linear regression | 5.84 ±0.74 |

## Code

For training linear regressions, the `train_ml_model.py` script contains one function for each brain feature.

For training DL models, see the `train_dl_model.py` script.
