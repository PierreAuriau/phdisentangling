import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict


class Encoder(nn.Module):

    def __init__(self, input_shape, channel=512, z_dim=256, kernel_size=4,
                 stride=2, padding=0, in_channels=1, bias=False):
        super(Encoder, self).__init__()

        self.z_dim = z_dim
        self.channel = channel

        self.conv1 = nn.Conv3d(in_channels, channel // 16, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.conv2 = nn.Conv3d(channel // 16, channel // 8, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm3d(channel // 8)
        self.conv3 = nn.Conv3d(channel // 8, channel // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.bn3 = nn.BatchNorm3d(channel // 4)
        self.conv4 = nn.Conv3d(channel // 4, channel // 2, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.bn4 = nn.BatchNorm3d(channel // 2)
        self.conv5 = nn.Conv3d(channel // 2, channel, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.bn5 = nn.BatchNorm3d(channel)

        output_shape = np.floor((np.array(input_shape) + 2*padding - kernel_size) / stride).astype(int) + 1
        for _ in range(4):
            output_shape = np.floor((output_shape + 2*padding - kernel_size) / stride).astype(int) + 1
        print(output_shape)
        out_features = int(channel*output_shape.prod())
        print(out_features)
        self.fc_representation = nn.Linear(out_features, z_dim, bias=False)

    def forward(self, x, _return_activations=False):
        batch_size = x.shape[0]

        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)

        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)

        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)

        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)

        # TODO : check if it is necessary
        h5 = F.leaky_relu(self.bn5(self.conv5(h4)), negative_slope=0.2)

        representation = self.fc_representation(h5.view(batch_size, -1))

        return representation


class Decoder(nn.Module):
    def __init__(self, channel: int = 512):
        super(Decoder, self).__init__()
        _c = channel

        self.fc = nn.Linear(256, 512 * 4 * 4 * 4)
        self.bn1 = nn.BatchNorm3d(_c)

        self.tp_conv2 = nn.ConvTranspose3d(channel, channel // 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channel // 2)

        self.tp_conv3 = nn.ConvTranspose3d(channel // 2, channel // 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(channel // 4)

        self.tp_conv4 = nn.ConvTranspose3d(channel // 4, channel // 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(channel // 8)

        self.tp_conv5 = nn.ConvTranspose3d(channel // 8, channel // 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm3d(channel // 16)

        self.tp_conv6 = nn.ConvTranspose3d(channel // 16, 1, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, latent):
        latent = latent.view(-1, 256)
        h = self.fc(latent)
        h = h.view(-1, 512, 4, 4, 4)
        h = F.relu(self.bn1(h))

        # h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv2(h)
        h = F.relu(self.bn2(h))

        # h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv3(h)
        h = F.relu(self.bn3(h))

        # h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv4(h)
        h = F.relu(self.bn4(h))

        # h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv5(h)
        h = F.relu(self.bn5(h))

        # h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv6(h)

        # h = torch.sigmoid(h)

        return h


class SimpleCNN(nn.Module):
    def __init__(self, z_dim: int = 256, encoder_channel: int = 512):
        super(CNN, self).__init__()
        self.encoder = Encoder(encoder_channel)
        self.decoder = Decoder()
        self.z_dim = z_dim

    def forward(self, x):
        representation = self.encoder(x)
        reconstruction = self.decoder(x)
        return representation, reconstruction


class CNN(nn.Module):
    def __init__(self, z_dim: int = 256, encoder_channel: int = 512, num_classes: int = 1,
                 input_shape: tuple = (128, 128, 128), in_channels=1, representation=False):

        super(CNN, self).__init__()
        self.encoder = Encoder(input_shape, channel=encoder_channel, z_dim=z_dim,
                               kernel_size=4, stride=2, padding=0, in_channels=in_channels, bias=False)
        self.z_dim = z_dim
        self.representation = representation
        self.relu = nn.ReLU()
        self.fc_classifier = nn.Linear(z_dim, num_classes)
        if num_classes == 1:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=-1)

    def forward(self, x):
        representation = self.encoder(x)
        if self.representation:
            return representation
        pred = self.fc_classifier(self.relu(representation))
        out = self.final_activation(pred)
        return out.squeeze(dim=1)


class SFCN(nn.Module):
    """https://www.frontiersin.org/articles/10.3389/fpsyt.2021.627996/full"""
    def __init__(self, channel_number=(32, 64, 128, 256, 256, 64), output_dim=40, dropout=True):
        super(SFCN, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('block_conv_%d' % i,
                                                  self.conv_block(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('block_conv_%d' % i,
                                                  self.conv_block(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))

        self.classifier = nn.Sequential()
        self.classifier.add_module('average_pool', nn.AvgPool3d(kernel_size=(4, 5, 4),
                                                                stride=None,
                                                                padding=1))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(p=0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_block(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        block = nn.Sequential(OrderedDict([
            ("conv", nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size)),
            ("batch_norm", nn.BatchNorm3d(out_channel))]))
        if maxpool:
            block.add_module("max_pool", nn.MaxPool3d(kernel_size=2, stride=maxpool_stride))
        block.add_module("relu", nn.ReLU())
        return block

    def forward(self, x):
        out = list()
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        x = F.log_softmax(x, dim=1)
        out.append(x)
        return out

