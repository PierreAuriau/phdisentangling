# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, latent_dim, activation="sigmoid", hidden_layer="full"):
        super(Classifier, self).__init__()
        self.latent_dim = latent_dim
        if hidden_layer == "half":
            self.fc = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim),
                        nn.ReLU(),
                        nn.Linear(self.latent_dim, self.latent_dim//2),
                        nn.ReLU(),
                        nn.Linear(self.latent_dim//2, 1))
        else:
            self.fc = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.latent_dim, self.latent_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.latent_dim, 1))
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation == nn.Softmax()
        else:
            self.activation = activation


    def forward(self, latent, return_logits=False):
        latent = latent.view(-1, self.latent_dim)
        h = self.fc(latent)
        if return_logits:
            return h
        else:
            pred = self.activation(h)
            return pred.squeeze()


class Projector(nn.Module):
    """Projector from 
    <https://github.com/facebookresearch/barlowtwins>
    """
    def __init__(self, latent_dim):
        super(Projector, self).__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(nn.Linear(self.latent_dim, 4*self.latent_dim, bias=False),
                                nn.BatchNorm1d(4*self.latent_dim),
                                nn.ReLU(),
                                nn.Linear(4*self.latent_dim, 4*self.latent_dim, bias=False),
                                nn.BatchNorm1d(4*self.latent_dim),
                                nn.ReLU(),
                                nn.Linear(4*self.latent_dim, 4*self.latent_dim, bias=False))

    def forward(self, latent):
        latent = latent.view(-1, self.latent_dim)
        out = self.fc(latent)
        return out


class Regressor(nn.Module):

    def __init__(self, latent_dim, num_classes=1, activation="sigmoid"):
        super(Regressor, self).__init__()
        self.latent_dim = latent_dim
        self.regressor = nn.Sequential(nn.Dropout(),
                                       nn.Linear(latent_dim, latent_dim//2),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(),
                                       nn.Linear(latent_dim//2, num_classes)
                                       )
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation == nn.Softmax()
        else:
            self.activation = activation
    
    def forward(self, latent, return_logits=False):
        latent = latent.view(-1, self.latent_dim)
        h = self.regressor(latent)
        if return_logits:
            return h
        else:
            pred = self.activation(h)
            return pred.squeeze()
