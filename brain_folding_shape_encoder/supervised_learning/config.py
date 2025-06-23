#!/usr/bin/env python

import os

class Config:
  def __init__(self):

    self.path_to_project = ""
    self.path_to_schemes = os.path.join(self.path_to_project, "data", "schemes")
    self.path_to_models = os.path.join(self.path_to_project, "models")
    self.path_to_data = os.path.join(self.path_to_project, "data", "processed")

    self.datasets = ("asd", "bd", "scz")
    self.splits = ("train", "validation", "internal_test", "external_test")
    
    self.n_embedding = 256
    self.architecture = "densenet"

    self.lr = 1e-4
    self.nb_epochs = 100
    self.weight_decay = 5e-3
    
    # SupCon parameters
    self.temperature = 0.1
    self.data_augmentation = "cutout"

    self.batch_size = 32
    self.num_workers = 8

    # ID types in participants dataframe
    self.id_types = {"participant_id": str,
                     "session": int,
                     "run": int,
                     "acq": int}
