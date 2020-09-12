import math
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.autograd import Variable

from .modules import initialize_filterbank

class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, shape, stride=stride, padding=shape // 2
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out

class Res2Dmp(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2):
        super(Res2Dmp, self).__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        out = x + out
        out = self.mp(self.relu(out))
        return out


class HarmonicSTFT(nn.Module):
    """HarmonicSTFT class"""
    def __init__(self,
                 sample_rate=16000,
                 n_fft=513,
                 win_length=None,
                 hop_length=None,
                 pad=0,
                 power=2,
                 normalized=False,
                 n_harmonic=6,
                 semitone_scale=2,
                 bw_Q=1.0,
                 learn_bw=None):
        """Instantiating HarmonicCNN class
            Args:
                sample_rate(float) : sampling rate
                n_fft(float) : number of fft bin
                win_length:int,
                hop_length:int,
                pad(int) : pad value of STFT,
                power(int) : Power of STFT,
                normalized(bool) : Normalize STFT,
                n_harmonic(int) : the number of n_harmonic
                semitone_scale(int) : a half step or a half tone, is the smallest musical interval
                bw_Q(int) : bandwidths initalizae,
                learn_bw(str) : learnable bandwidths
        """
        super(HarmonicSTFT, self).__init__()

        # Parameters
        self.sample_rate = sample_rate
        self.n_harmonic = n_harmonic
        self.bw_alpha = 0.1079
        self.bw_beta = 24.7

        # Spectrogram
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=None,
            pad=0,
            window_fn=torch.hann_window,
            power=power,
            normalized=normalized,
            wkwargs=None,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Initialize the filterbank. Equally spaced in MIDI scale.
        harmonic_hz, self.level = initialize_filterbank(
            sample_rate, n_harmonic, semitone_scale
        )

        # Center frequncies to tensor
        self.f0 = torch.tensor(harmonic_hz.astype("float32"))

        # Bandwidth parameters
        if learn_bw == "only_Q":
            self.bw_Q = nn.Parameter(torch.tensor(np.array([bw_Q]).astype("float32")))
        elif learn_bw == "fix":
            self.bw_Q = torch.tensor(np.array([bw_Q]).astype("float32"))

    def get_harmonic_fb(self):
        """get harmonic filter bank
        Returns:
            based on upslope and downslop, get triangle audiotory filter
        """
        bw = (self.bw_alpha * self.f0 + self.bw_beta) / self.bw_Q
        bw = bw.unsqueeze(0)  # (1, n_band)
        f0 = self.f0.unsqueeze(0)  # (1, n_band)
        fft_bins = self.fft_bins.unsqueeze(1)  # (n_bins, 1)

        up_slope = torch.matmul(fft_bins, (2 / bw)) + 1 - (2 * f0 / bw)
        down_slope = torch.matmul(fft_bins, (-2 / bw)) + 1 + (2 * f0 / bw)
        fb = torch.max(self.zero, torch.min(down_slope, up_slope))
        return fb

    def to_device(self, device: torch.cuda, n_bins: torch.Tensor):
        self.f0 = self.f0.to(device)
        self.bw_Q = self.bw_Q.to(device)
        # fft bins
        self.fft_bins = torch.linspace(0, self.sample_rate // 2, n_bins)
        self.fft_bins = self.fft_bins.to(device)
        self.zero = torch.zeros(1)
        self.zero = self.zero.to(device)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # stft
        spectrogram = self.spec(waveform)
        # to device
        self.to_device(waveform.device, spectrogram.size(1))
        # triangle filter
        harmonic_fb = self.get_harmonic_fb()
        harmonic_spec = torch.matmul(spectrogram.transpose(1, 2), harmonic_fb).transpose(1, 2)
        # (batch, channel, length) -> (batch, harmonic, f0, length)
        b, c, l = harmonic_spec.size()
        harmonic_spec = harmonic_spec.view(b, self.n_harmonic, self.level, l)
        harmonic_spec = self.amplitude_to_db(harmonic_spec)
        return harmonic_spec
