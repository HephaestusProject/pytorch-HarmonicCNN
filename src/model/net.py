# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from .ops import Conv_2d, HarmonicSTFT, Res2Dmp
from torch.autograd import Variable


class HarmonicCNN(nn.Module):
    """
    Won et al. 2020
    Data-driven harmonic filters for audio representation learning.
    Trainable harmonic band-pass filters.
    """
    def __init__(self,
                n_channels=128,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=128,
                n_class=50,
                n_harmonic=6,
                semitone_scale=2,
                learn_bw='only_Q'):
        """Instantiating HarmonicCNN class
        Args:
            n_channels(int) : number of channels
            sample_rate(float) : sampling rate
            n_fft(float) : number of fft bin
            f_min(float) : min frequency
            f_max(float) : max frequency
            n_class(int) : the number of classes
            n_harmonic(int) : the number of n_harmonic
            semitone_scale(int) : a half step or a half tone, is the smallest musical interval
            learn_bw(str) : learnable bandwidths
        """
        super(HarmonicCNN, self).__init__()

        # Harmonic STFT
        self.hstft = HarmonicSTFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_harmonic=n_harmonic,
            semitone_scale=semitone_scale,
            learn_bw=learn_bw,
        )
        self.hstft_bn = nn.BatchNorm2d(n_harmonic)

        # CNN
        self.layer1 = Conv_2d(n_harmonic, n_channels, pooling=2)
        self.layer2 = Res2Dmp(n_channels, n_channels, pooling=2)
        self.layer3 = Res2Dmp(n_channels, n_channels, pooling=2)
        self.layer4 = Res2Dmp(n_channels, n_channels, pooling=2)
        self.layer5 = Conv_2d(n_channels, n_channels*2, pooling=2)
        self.layer6 = Res2Dmp(n_channels*2, n_channels*2, pooling=(2,3))
        self.layer7 = Res2Dmp(n_channels*2, n_channels*2, pooling=(2,3))

        # Dense
        self.dense1 = nn.Linear(n_channels * 2, n_channels * 2)
        self.bn = nn.BatchNorm1d(n_channels * 2)
        self.dense2 = nn.Linear(n_channels * 2, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spectrogram
        x = self.hstft_bn(self.hstft(x))

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.sigmoid(x)

        return x
