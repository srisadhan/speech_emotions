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
                                    extract_MelSpectrogram)
from models.emotionet_conv import EMOTIONET, EMOTIONET2

import torch 
import torchaudio

import matplotlib.pyplot as plt 
import deepdish as dd
import argparse
from torch.utils.tensorboard import SummaryWriter
import librosa
import soundfile
import numpy as np

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

run_str = 'run'
with skip_run(run_str, 'Data_processing') as check, check():
    # import the data from .wav files and create a dictionary
    with skip_run(run_str, 'import_audio_files') as check, check():
        extract_using_torchaudio(config)
        
    # pad the audio files with zeros to make them of same size 
    with skip_run('skip', 'pad_zeros_to_waveforms_for_constant_length') as check, check():
        pad_zerosTo_waveforms(config)

    # NOTE: only run it if you want to neglect the change in intensities and copy in into repetitions
    with skip_run('skip', 'include_intensities_as_repetitions') as check, check():
        club_intensities_as_repetitions(config)

    with skip_run(run_str, 'extract_MelSpec_and_save') as check, check():
        extract_MelSpectrogram(config, True, False)
        # extract_MelSpectrogram(config, True, True)


# NOTE:  Only run this if you want to test the generated Mel-Spectrogram
with skip_run('skip', 'verify_MelSpec_conversion') as check, check():
        # compare the audio to verify if the conversion is right
        data = dd.io.load((Path(__file__).parents[1] / config['speech1_MelSpec']))
        
        for id in config['actors']:
            for emotion in config['emotions']:
                for intensity in config['intensities']:
                    for repetition in config['repetitions']:
                        # There are no available files for 02 intensity and 01 emotion
                        if (emotion == '01') and (intensity == '02'):
                            continue
                        elif emotion == '02':
                            specgram = data['Actor_' + id]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition]
                            print('Spectrogram size:', specgram.size())
                            
                            audio_clip = librosa.feature.inverse.mel_to_audio(specgram.data.numpy(),
                                                                sr=config['resampling_rate'],
                                                                n_fft=config['n_fft'],
                                                                hop_length=config['hop_length'],
                                                                win_length=config['win_length'])
                            
                            #NOTE: Use power
                            librosa.display.specshow(librosa.power_to_db(specgram.data.numpy(), ref=np.max),
                                    x_axis='s',
                                    y_axis='mel',
                                    sr=16000,
                                    fmax=8000,cmap='viridis')
                            plt.show()
                            
                            soundfile.write((str(Path(__file__).parents[1] / 'data/MelSpec_testSamples/actor_') + str(id) + 'emotion_' + str(emotion) + '.wav'),
                                        audio_clip,
                                        samplerate=config['resampling_rate'])  
                            sys.exit()     
                
                
               
######################################################
## Training an AutoEncoder
######################################################         
with skip_run('run', 'train_torch_model') as check, check():    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training the model on: {}".format(device))
    
    # NOTE: EMOTIONET takes 80x64 and EMOTIONET2 takes 80x128 
    loss_ = torch.nn.CrossEntropyLoss(reduction='sum')
    encoder = EMOTIONET2(dataset_train='speech1_MelSpec',
                        dataset_val='speech1_MelSpec',
                        device=device,
                        loss_=loss_,
                        load_model=False,
                        beta=config['beta']).to(device=device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    encoder.opt = optimizer

    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lr_lambda=lambda x: x*0.95)

    lr_scheduler = None
    encoder.train_loop(optimizer,
                        lr_scheduler,
                        loss_,
                        batch_size=encoder.config['batch_size'],
                        cpt=None)

with skip_run('skip', 'load_trained_torch_model') as check, check():    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training the model on: {}".format(device))
    
    loss_ = torch.nn.CrossEntropyLoss(reduction='sum')
    encoder = EMOTIONET2(dataset_train='speech1_MelSpec',
                        dataset_val='speech1_MelSpec',
                        device=device,
                        loss_=loss_,
                        load_model=True,
                        beta=config['beta']).to(device=device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    encoder.opt = optimizer
    cpt = 19000
    encoder.load_model_cpt(cpt=cpt, device=device)
    encoder.latent_sampling(visualize=True)
    
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lr_lambda=lambda x: x*0.95)
