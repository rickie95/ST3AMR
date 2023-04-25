import logging
import torch
import torchvision
import torch.nn as nn


class ConvolutionalAutoEncoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.enc_conv_layer_1 = nn.Conv2d(1, 16, 3, padding=1)
        self.enc_conv_layer_2 = nn.Conv2d(16, 4, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.dec_deconv_layer_1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.dec_deconv_layer_2 = nn.ConvTranspose2d(16, 1, 2, stride=2)


    def forward(self, features):
        x = torch.relu(self.enc_conv_layer_1(features))
        x = self.pool(x)

        x = torch.relu(self.enc_conv_layer_2(x))
        x = self.pool(x)

        x = torch.relu((self.dec_deconv_layer_1(x)))
        x = torch.sigmoid((self.dec_deconv_layer_2(x)))

        return x
