import torch
from torch import nn
from .deepunet import DeepUnet, DeepUnet0
from .constants import *
from .spec import MelSpectrogram
from .seq import BiGRU


class E2E(nn.Module):
    def __init__(self, hop_length, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1,
                 en_out_channels=16):
        super(E2E, self).__init__()
        self.mel = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX)
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * N_MELS, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid()
            )

    def forward(self, x):
        mel = self.mel(x.reshape(-1, x.shape[-1])).transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        # x = self.fc(x)
        hidden_vec = 0
        if len(self.fc) == 4:
            for i in range(len(self.fc)):
                x = self.fc[i](x)
                if i == 0:
                    hidden_vec = x
        return hidden_vec, x


class E2E0(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1,
                 en_out_channels=16):
        super(E2E0, self).__init__()
        self.unet = DeepUnet0(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * N_MELS, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid()
            )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x
