# coding: utf-8
import os

import numpy as np
import torch
import torchaudio
from torch.utils import data


class MTATDataset(data.Dataset):
    """MTATDataset class"""

    def __init__(self, root, split, input_length=80000):
        """Instantiating MTATDataset class
        Args:
            root (str): root directory
            split (str): train, valid, test dataset
        """
        self.root = root
        self.sampling_rate = 16000
        self.split = split
        self.input_length = input_length
        self.get_songlist()
        self.binary = np.load(os.path.join(self.root, "split", "binary.npy"))

    def get_songlist(self):
        if self.split == "TRAIN":
            self.fl = np.load(os.path.join(self.root, "split", "train.npy"))
        elif self.split == "VALID":
            self.fl = np.load(os.path.join(self.root, "split", "valid.npy"))
        elif self.split == "TEST":
            self.fl = np.load(os.path.join(self.root, "split", "test.npy"))
        else:
            print("Split should be one of [TRAIN, VALID, TEST]")

    def get_audio_tag(self, index):
        ix, fn = self.fl[index].split("\t")
        mp3_path = os.path.join(self.root, "mp3", fn)
        waveform, sr = torchaudio.load(mp3_path)
        downsample_resample = torchaudio.transforms.Resample(
            sr, self.sampling_rate, resampling_method="sinc_interpolation"
        )
        audio_tensor = downsample_resample(waveform)
        random_idx = int(
            np.floor(np.random.random(1) * (audio_tensor.shape[1] - self.input_length))
        )
        audio_tensor = audio_tensor[
            :, random_idx : random_idx + self.input_length
        ].squeeze(0)
        tag_binary = self.binary[int(ix)]
        return audio_tensor, tag_binary

    def __getitem__(self, index):
        audio_tensor, tag_binary = self.get_audio_tag(index)
        return audio_tensor.to(dtype=torch.float32), tag_binary.astype("float32")

    def __len__(self):
        return len(self.fl)


def get_audio_loader(root, batch_size, input_length, split="TRAIN", num_workers=0):
    data_loader = data.DataLoader(
        dataset=MTATDataset(root, split=split, input_length=input_length),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    return data_loader