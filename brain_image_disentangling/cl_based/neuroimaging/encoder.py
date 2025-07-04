# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports
import torch.nn as nn
from resnet import resnet18
from densenet import densenet121

class Encoder(nn.Module):
    def __init__(self, backbone, n_embedding):
        super(Encoder, self).__init__()
        self.latent_dim = n_embedding
        # encoder
        if backbone == "resnet18":
            self.encoder = resnet18(n_embedding=n_embedding, in_channels=1)
        elif backbone == "densenet121":
            self.encoder = densenet121(n_embedding=n_embedding, in_channels=1)
        # Add MLP projection.
        self.projector = nn.Sequential(nn.Linear(n_embedding, 2*n_embedding),
                                       nn.BatchNorm1d(2*n_embedding),
                                       nn.ReLU(),
                                       nn.Linear(2*n_embedding, n_embedding))

    def forward(self, x):
        representation = self.encoder(x)
        head = self.projector(representation)
        return representation, head
