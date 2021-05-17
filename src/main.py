import scipy as sp
from torch.optim import optimizer
import yaml
import sys
import wget 
import zipfile
import collections
import os 

from utils import skip_run
from pathlib import Path
from datasets.create_dataset import (download_dataset, extract_using_torchaudio,
                                    pad_zerosTo_waveforms, club_intensities_as_repetitions,
                                    extract_MelSpectrogram, constant_shaped_spectrogram, pooled_waveform_dataset)
from datasets.data_utils import gender_from_speaker_id_RAVDESS
from datasets.data_loaders import EmotionDataset
from models.speaker_encoder import SPEAKER_ENCODER
from models.emotionet_ae import EmotionNet_AE
from resemblyzer.voice_encoder import VoiceEncoder
import torch 
import torch.optim as optim

import matplotlib.pyplot as plt 
import deepdish as dd
import argparse
from torch.utils.tensorboard import SummaryWriter
import librosa
import soundfile
import numpy as np
import umap 


# The configuration file
config = yaml.load(open('src/config.yaml'), Loader=yaml.SafeLoader)

# Create the required folders
filepath = str(Path(__file__).parents[1] / 'data')
if not os.path.isdir(filepath):
    os.mkdir(filepath)

filepath = str(Path(__file__).parents[1] / 'data/raw')
if not os.path.isdir(filepath):
    os.mkdir(filepath)

filepath = str(Path(__file__).parents[1] / 'data/interim')
if not os.path.isdir(filepath):
    os.mkdir(filepath)

filepath = str(Path(__file__).parents[1] / 'data/processed')
if not os.path.isdir(filepath):
    os.mkdir(filepath)

######################################################
## RAVDESS dataset
######################################################
# download data from web and place it in the desired location
with skip_run('skip', 'download_RAVDESS_data') as check, check():
    url = config['url']
    download_dataset(url, config)


run_str = 'skip'
# Process the data and save the data in a dictionary format into the h5 file
with skip_run(run_str, 'Data_processing') as check, check():
    # import the data from .wav files and create a dictionary
    with skip_run(run_str, 'import_audio_files') as check, check():
        extract_using_torchaudio(config)
    
    with skip_run(run_str, 'extract_MelSpec_and_save') as check, check():
        extract_MelSpectrogram(config, intensity_flag=True, zero_pad_flag=False)
        # extract_MelSpectrogram(config, intensity_flag=True, zero_pad_flag=True)
    
    with skip_run(run_str, 'segment_speech_to_constant_length_MelSpec') as check, check():
        # load the previously extracted Mel Spectrograms
        data = dd.io.load((Path(__file__).parents[1] / config['speech1_MelSpec']))
        features, labels, speaker_id = constant_shaped_spectrogram(data, config, actors=config['train_actors'])
        Data = collections.defaultdict()
        Data['features'] = features
        Data['labels']   = labels
        Data['speaker_id'] = speaker_id
        dd.io.save(str(Path(__file__).parents[1] / config['const_MelSpec1']), Data)

        # load the previously extracted Mel Spectrograms
        data = dd.io.load((Path(__file__).parents[1] / config['speech2_MelSpec']))
        features, labels, speaker_id = constant_shaped_spectrogram(data, config, actors=config['train_actors'])
        Data = collections.defaultdict()
        Data['features'] = features
        Data['labels']   = labels
        Data['speaker_id'] = speaker_id
        dd.io.save(str(Path(__file__).parents[1] / config['const_MelSpec2']), Data)
        
    with skip_run(run_str, 'Pool the waveforms and save the dataset') as check, check():
        # load the waveform data
        data = dd.io.load(Path(__file__).parents[1] / config['speech1_data_raw'])
        features, labels, speaker_id = pooled_waveform_dataset(data, config, actors=config['train_actors'])
        Data = collections.defaultdict()
        Data['features'] = features
        Data['labels']   = labels
        Data['speaker_id'] = speaker_id
        dd.io.save(str(Path(__file__).parents[1] / config['pooled_waveforms1']), Data)
        
        data = dd.io.load(Path(__file__).parents[1] / config['speech2_data_raw'])
        features, labels, speaker_id = pooled_waveform_dataset(data, config, actors=config['train_actors'])
        Data = collections.defaultdict()
        Data['features'] = features
        Data['labels']   = labels
        Data['speaker_id'] = speaker_id
        dd.io.save(str(Path(__file__).parents[1] / config['pooled_waveforms2']), Data)
    
    # make the testing data using the left out speakers 
    with skip_run(run_str, 'Prepare the test data') as check, check():
            
        # Statement 1 and 2 melspectrograms
        data = dd.io.load((Path(__file__).parents[1] / config['speech1_MelSpec']))
        features, labels, speaker_id = constant_shaped_spectrogram(data, config, actors=config['test_actors'])
        Data = collections.defaultdict()
        Data['features'] = features
        Data['labels']   = labels
        Data['speaker_id'] = speaker_id
        dd.io.save(str(Path(__file__).parents[1] / config['const_test_MelSpec1']), Data)

        data = dd.io.load((Path(__file__).parents[1] / config['speech2_MelSpec']))
        features, labels, speaker_id = constant_shaped_spectrogram(data, config, actors=config['test_actors'])
        Data = collections.defaultdict()
        Data['features'] = features
        Data['labels']   = labels
        Data['speaker_id'] = speaker_id
        dd.io.save(str(Path(__file__).parents[1] / config['const_test_MelSpec2']), Data)
        
        # Statement 1 and 2 waveforms 
        reducer = umap.UMAP()
        encoder = VoiceEncoder()
                
        data = dd.io.load(Path(__file__).parents[1] / config['speech1_data_raw'])
        features, labels, speaker_id = pooled_waveform_dataset(data, config, actors=config['test_actors'])
        Data = collections.defaultdict()
                
        speaker_embedding = []    
        for i in range(len(features)):
            embeds = encoder.embed_utterance(features[i])
            speaker_embedding.append(embeds.reshape(1, -1))
        
        speaker_embedding = np.concatenate(speaker_embedding, axis=0)
        manifold_embedding = reducer.fit_transform(speaker_embedding)
        
        Data = {'speaker_embeds': speaker_embedding,
                'manifold_embeds': manifold_embedding,
                'speaker_id': speaker_id}
        dd.io.save(str(Path(__file__).parents[1] / config['umap_resemblyzer_test1']), Data)

    
        data = dd.io.load(Path(__file__).parents[1] / config['speech2_data_raw'])
        features, labels, speaker_id = pooled_waveform_dataset(data, config, actors=config['test_actors'])
        
        speaker_embedding = []    
        for i in range(len(features)):
            embeds = encoder.embed_utterance(features[i])
            speaker_embedding.append(embeds.reshape(1, -1))
        
        speaker_embedding = np.concatenate(speaker_embedding, axis=0)
        manifold_embedding = reducer.fit_transform(speaker_embedding)
        
        Data = {'speaker_embeds': speaker_embedding,
                'manifold_embeds': manifold_embedding,
                'speaker_id': speaker_id}
        dd.io.save(str(Path(__file__).parents[1] / config['umap_resemblyzer_test2']), Data)
    #--------------------------------------------------------------------------------------#          
               

######################################################
## Speaker Encoder - Chaitanya's trained model
###################################################### 
with skip_run('skip', 'Project the Speaker Embeddings to a Manifold') as check, check():
    data = dd.io.load(str(Path(__file__).parents[1] / config['pooled_waveforms1']))

    speaker_id = data['speaker_id']
    reducer = umap.UMAP()
    
    model = SPEAKER_ENCODER(load_model=True,
                            epoch=0,
                            device=torch.device('cpu'),
                            loss_=None, mode='validate')
    
    model.load_model_cpt(cpt=13, device=torch.device('cpu'))

    model.eval()
    
    speaker_embedding = []
    
    for i in range(len(data['features'])):
        embeds, _ = model.embed(data['features'][i], group=True)
        speaker_embedding.append(embeds.reshape(1, -1))
    
    speaker_embedding = np.concatenate(speaker_embedding, axis=0)
    manifold_embedding = reducer.fit_transform(speaker_embedding)
    
    Data = {'speaker_embeds': speaker_embedding,
            'umap_embeds' : manifold_embedding,
            'speaker_id': speaker_id}
    dd.io.save(str(Path(__file__).parents[1] / config['umap_speaker_embeddings']), Data)

with skip_run('skip', 'Plot the UMAP embeddings')as check, check():
    data = dd.io.load(str(Path(__file__).parents[1] / config['umap_speaker_embeddings']))
    speaker_id = data['speaker_id'].detach().numpy()
    manifold_embedding = data['umap_embeds']
    
    gender_id = gender_from_speaker_id_RAVDESS(config, speaker_id)
    
    female_ind = np.multiply(np.arange(0, gender_id.shape[0]), gender_id.tolist())
    male_ind   = np.multiply(np.arange(0, gender_id.shape[0]), np.subtract(np.ones(gender_id.shape, dtype=np.int), gender_id, dtype=np.int).tolist())

    plt.figure()
    plt.scatter(manifold_embedding[:, 0], manifold_embedding[:, 1], c=gender_id, cmap='viridis')
    plt.colorbar()
    plt.title('Male and Female Speaker embeddings on the UMAP manifold')
    
    plt.figure()
    plt.scatter(manifold_embedding[male_ind, 0], manifold_embedding[male_ind, 1], c=speaker_id[male_ind], cmap='viridis')
    plt.colorbar()
    plt.title('Male Speaker embeddings on the UMAP manifold')
    
    plt.figure()
    plt.scatter(manifold_embedding[female_ind, 0], manifold_embedding[female_ind, 1], c=speaker_id[female_ind], cmap='viridis')
    plt.colorbar()
    plt.title('Female Speaker embeddings on the UMAP manifold')
    
    plt.show()
    
######################################################
## Voice Encoder - from Resemblyzer project
###################################################### 
with skip_run('skip', 'Project the Voice Embeddings to a Manifold') as check, check():
    reducer = umap.UMAP()
    encoder = VoiceEncoder()
    
    # For statement 1
    data = dd.io.load(str(Path(__file__).parents[1] / config['pooled_waveforms1']))
    speaker_id = data['speaker_id']
    speaker_embedding = []    
    for i in range(len(data['features'])):
        embeds = encoder.embed_utterance(data['features'][i])
        speaker_embedding.append(embeds.reshape(1, -1))
    
    speaker_embedding = np.concatenate(speaker_embedding, axis=0)
    manifold_embedding = reducer.fit_transform(speaker_embedding)
    
    Data = {'speaker_embeds': speaker_embedding,
            'manifold_embeds': manifold_embedding,
            'speaker_id': speaker_id}
    dd.io.save(str(Path(__file__).parents[1] / config['umap_resemblyzer_embeds1']), Data)
    
    
    # For statement 2
    data = dd.io.load(str(Path(__file__).parents[1] / config['pooled_waveforms1']))
    speaker_id = data['speaker_id']
    speaker_embedding = []    
    for i in range(len(data['features'])):
        embeds = encoder.embed_utterance(data['features'][i])
        speaker_embedding.append(embeds.reshape(1, -1))
    
    speaker_embedding = np.concatenate(speaker_embedding, axis=0)
    manifold_embedding = reducer.fit_transform(speaker_embedding)
    
    Data = {'speaker_embeds': speaker_embedding,
            'manifold_embeds': manifold_embedding,
            'speaker_id': speaker_id}
    dd.io.save(str(Path(__file__).parents[1] / config['umap_resemblyzer_embeds2']), Data)

with skip_run('skip', 'Plot the UMAP embeddings')as check, check():
    data = dd.io.load(str(Path(__file__).parents[1] / config['umap_resemblyzer_embeds1']))
    speaker_id = data['speaker_id'].detach().numpy()
    manifold_embedding = data['speaker_embeds']
    
    gender_id = gender_from_speaker_id_RAVDESS(config, speaker_id)
    
    female_ind = np.multiply(np.arange(0, gender_id.shape[0]), gender_id.tolist())
    male_ind   = np.multiply(np.arange(0, gender_id.shape[0]), np.subtract(np.ones(gender_id.shape, dtype=np.int), gender_id, dtype=np.int).tolist())

    plt.figure()
    plt.scatter(manifold_embedding[:, 0], manifold_embedding[:, 1], c=gender_id, cmap='viridis')
    plt.colorbar()
    plt.title('Male and Female Speaker embeddings on the UMAP manifold')
    
    plt.figure()
    plt.scatter(manifold_embedding[male_ind, 0], manifold_embedding[male_ind, 1], c=speaker_id[male_ind], cmap='viridis')
    plt.colorbar()
    plt.title('Male Speaker embeddings on the UMAP manifold')
    
    plt.figure()
    plt.scatter(manifold_embedding[female_ind, 0], manifold_embedding[female_ind, 1], c=speaker_id[female_ind], cmap='viridis')
    plt.colorbar()
    plt.title('Female Speaker embeddings on the UMAP manifold')
    
    plt.show()
    
######################################################
## Train EmotionNet
###################################################### 

with skip_run('run', 'Train the Emotion autoencoder') as check, check():
    datapath = str(Path(__file__).parents[1] / config['const_MelSpec1'])
    embeds_path = str(Path(__file__).parents[1] / config['umap_speaker_embeddings'])
    
    dataset = EmotionDataset(datapath, embeds_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Train the model on: {}'.format(device))
    
    model = EmotionNet_AE(device=device)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # scheduler = None
    
    model.train_model(dataloader, optimizer, device, lr_scheduler=scheduler, separate_speaker=True)
    

with skip_run('run', "UMAP emotion embeddings with trained model") as check, check():
    
    # Initialize the model    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Train the model on: {}'.format(device))
    
    model = EmotionNet_AE(device=device)
    model.to(device)
    
    checkpoint = torch.load(str(Path(__file__).parents[1] / 'trained_models/EmotionNet_AE/run_with_speaker/EmotionNet_AE_Epoch_7984.pt'))
    
    
    # Test with speech 1 left out actors data
    datapath = str(Path(__file__).parents[1] / config['const_test_MelSpec1'])
    embeds_path = str(Path(__file__).parents[1] / config['umap_resemblyzer_test1'])
    dataset = EmotionDataset(datapath, embeds_path)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    model.validate_model(dataloader, checkpoint=checkpoint)
    
    # Test with speech 2 left out actors data
    datapath = str(Path(__file__).parents[1] / config['const_test_MelSpec2'])
    embeds_path = str(Path(__file__).parents[1] / config['umap_resemblyzer_test2'])
    dataset = EmotionDataset(datapath, embeds_path)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    model.validate_model(dataloader, checkpoint=checkpoint)
    
    
    # Test with speech 2 all the actors data 
    datapath = str(Path(__file__).parents[1] / config['const_MelSpec2'])
    embeds_path = str(Path(__file__).parents[1] / config['umap_resemblyzer_embeds2'])
    dataset = EmotionDataset(datapath, embeds_path)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    model.validate_model(dataloader, checkpoint=checkpoint)
    plt.show()