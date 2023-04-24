import logging
import torch
import torchvision
import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_input_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_hidden_layer_1 = nn.Linear(
            in_features=128, out_features=64
        )
        self.encoder_hidden_layer_2 = nn.Linear(
            in_features=64, out_features=32
        )
        self.decoder_hidden_layer_1 = nn.Linear(
            in_features=32, out_features=64
        )
        self.decoder_hidden_layer_2 = nn.Linear(
            in_features=64, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_input_layer(features)
        activation = torch.relu(activation)
        activation = self.encoder_hidden_layer_1(activation)
        activation = torch.relu(activation)
        code = self.encoder_hidden_layer_2(activation)

        code = torch.sigmoid(code)

        activation = self.decoder_hidden_layer_1(code)
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer_2(activation)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        return reconstructed
