import sys
import os
#nopep8
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

from yaml import safe_load
import pandas as pd
from collections import Counter
import librosa
from librosa.display import specshow
import numpy as np
import webrtcvad
import soundfile
from scipy.ndimage import binary_dilation
import pathlib
from colorama import Fore
from tqdm import tqdm
import warnings
import h5py
import matplotlib.pyplot as plt
from torch.utils import data
import torch
from random import randint, sample
from yaml import safe_load
import deepdish as dd
import torch.nn.functional as F

with open('src/config.yaml', 'r') as f:
    config = safe_load(f.read())


class HDF5TorchDataset(data.Dataset):
    def __init__(self, emotion_data, device=torch.device('cpu')):
        self.hdf5_file = os.path.join(config['interim_data_dir'],
                                      '{}.h5'.format(emotion_data))
        self.hdf5_file_obj = h5py.File(self.hdf5_file, 'r')
        self.actors = list(self.hdf5_file_obj.keys())
        with open('src/config.yaml', 'r') as f:
            self.config = safe_load(f.read())
        self.device = device

    def __len__(self):
        # FIXME: What should be the size here?
        # return len(self.actors) * len(config['emotions']) * len(config['repetitions'])
        return len(self.actors)  + int(1e2)
    
    def __get_dd_paths__(self, actors, intensity, emotion):

        emotions_paths_src = [
            os.path.join(
                '/', actor, 'emotion_{}'.format(sample(emotion, 1)[0]),
                'intensity_{}'.format(intensity[0]),
                'repete_{}'.format(sample(config['repetitions'], 1)[0]))
            for actor in actors
        ]
        return emotions_paths_src

    def __get_rand_segs__(self, emotions):
        emotions_rand = []
        for emotion in emotions:
            if emotion.shape[1] < self.config['mel_seg_length']:
                plen = (self.config['mel_seg_length'] - emotion.shape[1]) // 2 +2
                emotion = F.pad(emotion.unsqueeze(0), (plen, plen), mode='reflect').squeeze()
            rix = randint(0, emotion.shape[1] - self.config['mel_seg_length'])
            emotion = emotion[:, rix:rix + config['mel_seg_length']]
            emotions_rand.append(emotion)
        return torch.stack(emotions_rand).squeeze().to(device=self.device)

 
    def __getitem__(self,ix=0):
        rand_actor = sample(self.hdf5_file_obj.keys(), 1)
        rand_emotion = sample(config['emotions'], 1)
        rand_intensity = sample(config['intensities'], 1)
        
        if rand_emotion[0] == '01':
            rand_intensity = config['intensities']

        emotion_path = self.__get_dd_paths__(rand_actor,rand_intensity,rand_emotion)
        emotions = dd.io.load(self.hdf5_file, emotion_path)
        emotion = self.__get_rand_segs__(emotions)
        # print(emotion.size())
        # for 
        
        return emotion.to(device=self.device)

if __name__ == "__main__":
    hdf5 = HDF5TorchDataset('speech1_MelSpec')
    loader = data.DataLoader(hdf5,batch_size=3)
    for i, x in enumerate(loader):
        print(x.shape)
        S_dB = librosa.power_to_db(x[0], ref=np.max)
        librosa.display.specshow(S_dB,
                                 x_axis='s',
                                 y_axis='mel',
                                 sr=16000,
                                 fmax=8000,cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.show()
        print(i, x.squeeze().shape)