import sys
import os
from numpy.core.defchararray import index
from scipy.ndimage.measurements import label
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
import random 
from yaml import safe_load
import deepdish as dd
import torch.nn.functional as F
from pathlib import Path
from src.datasets.create_dataset import fix_length_melspectrogram_RAVDESS, pool_melspectrogram_RAVDESS

with open('src/config.yaml', 'r') as f:
    config = safe_load(f.read())
   
#########################################################
class EmotionDataset(data.Dataset):
    """Emotion Dataset loader"""
    def __init__(self, emotion_data):
        super(EmotionDataset).__init__()
        
        if isinstance(emotion_data, str):
            self.path = emotion_data
            self.features, self.embeds, self.speaker_id, self.labels = self.import_data()
        elif isinstance(emotion_data, dict):
            self.features, self.embeds, self.speaker_id, self.labels = emotion_data['features'], emotion_data['speaker_embeds'], emotion_data['speaker_id'], emotion_data['labels']
        
    def import_data(self):
        data = dd.io.load(self.path)

        features = data['features']
        embeds = data['speaker_embeds']
        labels = data['labels']
        speaker_id = data['speaker_id']
        return features, embeds, speaker_id, labels
        
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
                   'labels': torch.tensor(labels, dtype=torch.long),
                   'speaker_id':torch.tensor(self.speaker_id, dtype=torch.long)}
        return samples

class EmotionEncoderDataset(data.Dataset):
    """Emotion Dataset loader"""
    def __init__(self, emotion_data, emotions_per_batch, utterances_per_emotion, mode='train'):
        super(EmotionEncoderDataset).__init__()
        
        self.mode = mode
        if isinstance(emotion_data, str):
            self.path = emotion_data
            self.features, self.embeds, self.speaker_id, self.labels = self.import_data()
        elif isinstance(emotion_data, dict):
            self.features, self.embeds, self.speaker_id, self.labels = emotion_data['features'], emotion_data['speaker_embeds'], emotion_data['speaker_id'], emotion_data['labels']
        
        self.emotions_per_batch = emotions_per_batch
        self.utterances_per_emotion = utterances_per_emotion
        self.emotion_categories = np.sort(np.unique(self.labels.reshape(-1,)))
        self.n_emotions = len(self.emotion_categories.tolist())

        if self.mode == 'train':
            self.refactor_data()
            self.create_data_bins()
        
    def import_data(self):
        data = dd.io.load(self.path)

        features = data['features']
        embeds = data['speaker_embeds']
        labels = data['labels']
        speaker_id = data['speaker_id']

        return features, embeds, speaker_id, labels

    def refactor_data(self):
        """ Refactor the data into emotion categories
        features - (emotion_per_batch x utterances x n_mels x utterance_length)
        embeds - (emotion_per_batch x utterances x 1 x embedding_size)
        speaker_id - (emotion_per_batch x utterances)
        labels - (emotion_per_batch x utterances)
        """
        emotion_labels = self.labels.reshape(-1,)
        ind = np.arange(len(emotion_labels))
        
        # sort the data according to emotion categories 
        emo_indices = []
        for emo in self.emotion_categories:
            emo_ind_i = ind[emotion_labels == emo]
            random.shuffle(emo_ind_i)
            emo_indices.append(emo_ind_i)
        emo_indices = np.array(emo_indices)
        
        self.features   = self.features[emo_indices]
        self.embeds     = self.embeds[emo_indices]
        self.speaker_id = self.speaker_id[emo_indices]
        self.labels     = self.labels[emo_indices]

        self.n_utterances = self.features.shape[1]
        self.n_bins = int(np.ceil(self.n_emotions / self.emotions_per_batch))

    def create_data_bins(self):
        # create bins of emotions 
        features, embeds, speaker_id, labels = [], [], [], []
        for i in range(0, self.features.shape[0], self.emotions_per_batch):
            for j in range(0, self.n_utterances, self.utterances_per_emotion):
                features.append(self.features[i:i+self.emotions_per_batch, j:j+self.utterances_per_emotion, :, :])
                embeds.append(self.embeds[i:i+self.emotions_per_batch, j:j+self.utterances_per_emotion, :, :])
                labels.append(self.labels[i:i+self.emotions_per_batch, j:j+self.utterances_per_emotion])
                speaker_id.append(self.speaker_id[i:i+self.emotions_per_batch, j:j+self.utterances_per_emotion])
        
        self.features   = features
        self.embeds     = embeds
        self.labels     = labels
        self.speaker_id = speaker_id

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()              
        samples = {'features': torch.tensor(self.features[idx], dtype=torch.float32, requires_grad=True),
                'speaker_embeds': torch.tensor(self.embeds[idx], dtype=torch.float32), 
                'labels': torch.tensor(self.labels[idx], dtype=torch.long),
                'speaker_id':torch.tensor(self.speaker_id[idx], dtype=torch.long)}
    
        return samples
    
def griffin_lim_aud(self, spec, emotion, save_audio=False):
    """ Generate audio samples from waveforms using Griffin approach
    """
    if config['use_logMel']:
        spec = librosa.db_to_power(spec.detach().numpy())
    else:
        spec = spec.detach().numpy()
        
    audio = librosa.feature.inverse.mel_to_audio(spec,
                                        sr=config['resampled_rate'],
                                        n_fft=config['n_fft'],
                                        hop_length=config['hop_length'],
                                        win_length=config['win_length'])

    if save_audio:
        savepath = os.path.join(os.getcwd(), 'emotion_{}.wav'.format(emotion))
        soundfile.write(savepath, audio, samplerate=config['resampled_rate'])
    
if __name__ == "__main__":
    filename = 'RAVDESS_simple_40melspec'
    data = dd.io.load((Path(__file__).parents[2] / config[filename]))
    
    Data = pool_melspectrogram_RAVDESS(data, config, actors=config['train_actors'])

    dataset = EmotionEncoderDataset(Data, emotions_per_batch=5, mode='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    
    for i, data in enumerate(dataloader):
        print(data['features'].shape)
    
    sys.exit()
        
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