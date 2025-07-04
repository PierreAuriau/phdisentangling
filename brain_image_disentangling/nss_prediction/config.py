# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os

class Config:

    def __init__(self):
        # Data
        self.study_dir = "/neurospin/psy_sbox/AUSZ"

        # Project
        self.path2analyse = "/neurospin/psy_sbox/analyses/2023_pauriau_EarlyBrainMarkersWithContrastiveAnalysis"
        # data
        self.path2data = os.path.join(self.path2analyse, "data")
        self.path2raw = os.path.join(self.path2data, "raw")
        self.path2interim = os.path.join(self.path2data, "interim")
        self.path2processed = os.path.join(self.path2data, "processed")
        self.path2schemes = os.path.join(self.path2data, "schemes")
        self.path2brainmasks = os.path.join(self.path2data, "brain_masks")
        # models
        self.path2models = os.path.join(self.path2analyse, "models")
        # figures
        self.path2figures = os.path.join(self.path2analyse, "figures")
 
        self.home_local = "/home_local/pa267054/ausz"
        
        # Data type of participant dataframe columns
        self.id_types = {"participant_id": str,
                         "session": int,
                         "acq": int,
                         "run": int}
        
        # Parameters for stratification
        self.unique_keys = ["participant_id", "session", "study"]
        self.nb_folds = 10
        self.stratify = ["NSS"]
        
        # Label mapping when predicting categorical phenotypes (sex and diagnosis)
        self.label_mapping = {"M": 0,
                              "m": 0,
                              "F": 1,
                              "control": 0,
                              "scz": 1,
                              "scz-asd": 1,
                              "asd": 1}

