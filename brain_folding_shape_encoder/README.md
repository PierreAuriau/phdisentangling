# Brain folding shape encoder

In this folder, we built an encoder to extract meaningful representations from brain folding shapes. We evaluated the learned representations on three case-control classification tasks:
* Healthy Control (HC) vs Autism Spectrum Disorders (ASD)
* Healthy Control (HC) vs Bipolar disorders (BD)
* Healthy Control (HC) vs Schizophrenia (SCZ)

We compared three approaches:
* [supervised_learning](./supervised_learning): training the model from scratch in a supervised setting
* [transfer_learning](./transfer_learning): pre-training of the model on a large cohort (UK Biobank)
* [regional_approach](./regional_approach): aggregation of local experts on a specific brain area (based on the brain folding foundation model Champollion)

These codes reproduce the experiments in Chapter 2 of my thesis manuscript.
___

## Datasets

The 3 clinical datasets are derived mostly from public cohorts excepted for 
BIOBD, BSNIP1 and PRAGUE, that are private for clinical research. The three datasets are described below:

**Dataset** | **# Subjects** | **Age** (avg±std) | **Sex (\%F)** | **# Sites** | **Studies**
| :---:| :---: | :---: | :---: | :---: | :---: |
HC<br>SCZ | 761<br>532 | 33 ± 12<br>34 ± 12 | 51<br>29 | 12 | [BSNIP1](http://b-snip.org), [CANDI](https://www.nitrc.org/projects/candi_share), [CNP](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5664981/),   PRAGUE, [SCHIZCONNECT](http://schizconnect.org)
HC<br>BD | 695<br>469 | 37 ± 14<br>39 ± 12 | 54<br>57 | 15 | [BIOBD](https://pubmed.ncbi.nlm.nih.gov/29981196/), [BSNIP1](http://b-snip.org), [CANDI](https://www.nitrc.org/projects/candi_share), [CNP](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5664981/)
HC<br>ASD | 926<br>813 | 16 ± 9<br>16 ± 9 | 25<br>13 | 30 | [ABIDE I](http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html) , [ABIDE II](http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html)

## Results

The ROC AUC scores average over the 10-fold cross-validation are reported in the table below:

| Task | Test set | Supervised learning | Transfer learning | Regional approach |
| --- | --- | :---: | :---: | :---: |
| HC vs ASD | internal | 53.4 ± 3.4 |  63.0 ± 1.1 |  58.3 ± 2.8 |
|           | external | 53.4 ± 5.4 | 66.5 ± 1.0 |  56.7 ± 3.2 |
| HC vs BD | internal | 56.2 ± 4.8 | 68.7 ± 1.3 | 61.1 ± 2.7 |
|          | external | 52.7 ± 2.2 |  59.7 ± 1.5 |  59.3 ± 1.4 |
| HC vs SCZ | internal | 63.9 ± 6.5 | 66.3 ± 0.6 |  63.0 ± 1.4 |
|           | external | 53.2 ± 3.6 |  57.6 ± 0.4 |  52.9 ± 2.2 |

## Code

To train a model:
``` bash
# create virtual environment
python3 -m venv myenv
# install librairies
python3 -m pip install -r requirements.txt
# edit configuration to update path and parameters
nano config.py
# train model
python train.py --help
```

In each folder, the script are organized as follows:
* ```config.py```: path to directories and default values for parameters
* ```dataset.py```: dataset to load data
* ```datamanager.py```: manager that load dataset correctly
* ```data_augmentation.py```: data augmentations for the pre-training
* ```loss.py```: implementation of the BarlowTwins loss for the original paper
* ```model.py```: torch module with fit and test method
* ```train.py```: script to load data and train models
* ```test.py```: script to test models
* ```explain.py```: patch occlusion XAI method
* ```log.py```: function to set up logs and logger class
* ```classifier.py mlp.py densenet.py resnet.py alexnet.py```: deep neural network architectures
* ```make_dataset.py```: create array and dataframe from Champollion embeddings
* ```make_pca.py```: make a dimension reduction of Champollion embeddings with an ACP

