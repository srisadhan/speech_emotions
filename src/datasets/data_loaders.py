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
        return len(self.actors)  + int(1e3)
    
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


#####################################

class HDF5TorchDataset2(data.Dataset):
    def __init__(self, emotion_data_path, config, device=torch.device('cpu')):

        self.hdf5_file =  emotion_data_path
    
        self.config = config
        self.device = device

    def __len__(self):
        return len(config['actors']) * (len(config['emotions']) * len(config['repetitions']) * len(config['intensities']) - 2)
     
    def __getitem__(self,ix=0):        
        data = dd.io.load(self.hdf5_file)
        features, labels = data['features'], data['labels']
        
        # return features.to(device=self.device), labels.to(device=self.device)
        return features[ix].to(device=self.device)
    
#########################################################
class EmotionDataset(data.Dataset):
    """Emotion Dataset loader"""
    def __init__(self, emotion_data_path, embeddings_path):
        super(EmotionDataset).__init__()
        
        self.path = emotion_data_path
        self.embed_path = embeddings_path
        self.features, self.embeds, self.labels = self.import_data()
        
    def import_data(self):
        data = dd.io.load(self.path)
        embeds = dd.io.load(self.embed_path)

        features = data['features']
        labels = data['labels']
        embeds = embeds['speaker_embeds']

        return features, embeds, labels
        
    def __len__(self):
        return (self.features).shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if len(self.labels) > 0:
            labels = self.labels[idx]
        else:
            labels = []
            
        samples = {'features': torch.tensor(self.features[idx, :, :], dtype=torch.float32, requires_grad=True),
                   'speaker_embeds': torch.tensor(self.embeds[idx, :], dtype=torch.float32), 
                   'labels': torch.tensor(labels, dtype=torch.long)}
        return samples
     
if __name__ == "__main__":
    hdf5 = HDF5TorchDataset('speech1_MelSpec')
    loader = data.DataLoader(hdf5,batch_size=3)
    for i, x in enumerate(loader):
        print(x.shape)
        
        if config['use_logMel']:
            S_dB = librosa.power_to_db(x[0])
        else:
            S_dB = x[0]
            
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